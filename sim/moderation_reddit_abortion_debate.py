import argparse
import json
import time
from pathlib import Path
from typing import Any, NotRequired, TypedDict
from uuid import uuid4

import torch
from huggingface_hub import snapshot_download
from openai import OpenAI
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


class Comment(TypedDict):
    id: str
    author: str
    body: str
    created_utc: float
    replies: list["Comment"]
    round_idx: NotRequired[int]
    toxicity: NotRequired[float]


class RedditThread(TypedDict):
    submission_id: str
    author: str
    title: str
    selftext: NotRequired[str]
    created_utc: float
    comments: list[Comment]
    body_toxicity: NotRequired[float]


class AlignmentProfile(TypedDict):
    author: str
    name: str
    persona: str
    observed_comments: list[Comment]


class ReasoningResult(TypedDict):
    explanation: str
    category: str
    issue_type: str
    points: int


class InterventionPayload(TypedDict):
    strategy: str
    target: str
    tone_used: str
    rationale: str
    intervention_text: str


def response_content(response: Any) -> str:
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if message is not None:
        return str(getattr(message, "content", "") or "")

    delta = getattr(choice, "delta", None)
    if delta is not None:
        return str(getattr(delta, "content", "") or "")

    return ""


def extract_json_dict(raw_text: str) -> dict[str, Any] | None:
    candidates = [raw_text.strip()]
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw_text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_reasoning_result(raw_text: str) -> ReasoningResult:
    parsed = extract_json_dict(raw_text)
    if parsed is None:
        return {
            "explanation": (
                "Model output was malformed; applied a conservative fallback "
                "classification for a generated comment."
            ),
            "category": "toxic",
            "issue_type": "unclassified",
            "points": 3,
        }

    return {
        "explanation": str(parsed.get("explanation", "")),
        "category": str(parsed.get("category", "toxic")),
        "issue_type": str(parsed.get("issue_type", "unclassified")),
        "points": int(parsed.get("points", 3)),
    }


def parse_intervention_payload(raw_text: str, issue_type: str) -> InterventionPayload:
    parsed = extract_json_dict(raw_text)
    if parsed is None:
        return fallback_intervention(issue_type)

    required = {"strategy", "target", "tone_used", "rationale", "intervention_text"}
    if not required.issubset(parsed):
        return fallback_intervention(issue_type)

    return {
        "strategy": str(parsed["strategy"]),
        "target": str(parsed["target"]),
        "tone_used": str(parsed["tone_used"]),
        "rationale": str(parsed["rationale"]),
        "intervention_text": str(parsed["intervention_text"]),
    }


class ToxicityClassifier:
    """Local CGA classifier copied from models.py for this standalone script."""

    def __init__(self, model_path: str | Path = "models/cga_deberta_onnx_int8") -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            snapshot_download(
                repo_id="lbux/cga-deberta-onnx-int8",
                local_dir=self.model_path,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.ort_model = ORTModelForSequenceClassification.from_pretrained(
            self.model_path,
            file_name="model_quantized.onnx",
        )

    def predict(self, text: str) -> float:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.ort_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return float(probs[0][1].item())


class RedditDebateAgent:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        stream: bool,
        thinking: bool,
        topic: str,
        name: str,
        persona: str,
        aligned_author: str,
        observed_comments: list[Comment],
        max_context_turns: int = 10,
    ) -> None:
        self.client = client
        self.model = model
        self.stream = stream
        self.thinking = thinking
        self.topic = topic
        self.name = name
        self.persona = persona
        self.aligned_author = aligned_author
        self.observed_comments = observed_comments
        self.max_context_turns = max_context_turns

    def speak(self, history: list[dict[str, Any]], submission: RedditThread) -> str:
        chat_messages = self._build_chat_messages(history, submission)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            stream=self.stream,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
        )

        if self.stream is False:
            text = response_content(response).strip()
            print(f"{self.name} replies: {text}\n")
            return text

        return self._stream_response(response)

    def _build_chat_messages(
        self, history: list[dict[str, Any]], submission: RedditThread
    ) -> list[dict[str, str]]:
        recent_history = history[-self.max_context_turns :]
        history_prompt = "\n\n".join(
            f"{idx}. u/{turn['author']} ({turn['id']}):\n{turn['body']}"
            for idx, turn in enumerate(recent_history, start=1)
        )

        submission_context = (
            f"Reddit submission title: {submission.get('title', '').strip()}\n"
            f"Submission URL: {submission.get('submission_url', '')}\n"
            f"Original post body:\n{(submission.get('selftext') or '').strip()}"
        )
        alignment_evidence = self._format_alignment_evidence()

        return [
            {
                "role": "system",
                "content": (
                    f"You are {self.name}, an AI debate participant aligned with "
                    f"the observed Reddit user u/{self.aligned_author}. You are "
                    f"participating in a Reddit thread debate about {self.topic}. "
                    f"Your alignment goal: {self.persona} Do not claim to be the "
                    "real Reddit user. Write as a simulated Reddit participant "
                    "whose stance, arguments, and tone stay consistent with the "
                    "observed user."
                ),
            },
            {
                "role": "user",
                "content": f"""
{submission_context}

Thread context, from older to newer:
{history_prompt}

Observed comments from the Reddit user you are aligned with:
{alignment_evidence}

Task:
Reply to the latest comment in this Reddit thread.
Use the real thread context above.
Stay aligned with the observed user's stance, recurring arguments, and tone.
Be direct, conversational, and civil.
Do not invent citations or personal experiences.
Do not add a username, heading, or label.
Keep the reply within 120 words.
""",
            },
        ]

    def _format_alignment_evidence(self) -> str:
        return "\n\n".join(
            f"{idx}. u/{comment['author']} ({comment['id']}):\n{comment['body']}"
            for idx, comment in enumerate(self.observed_comments, start=1)
        )

    def _stream_response(self, response: Any) -> str:
        parts: list[str] = []
        print(f"{self.name}: ", end="", flush=True)
        for chunk in response:
            content = response_content(chunk)
            if content:
                print(content, end="", flush=True)
                parts.append(content)
        print()
        return "".join(parts).strip()


class ModerationReasoner:
    def __init__(self, client: OpenAI, model: str, thinking: bool) -> None:
        self.client = client
        self.model = model
        self.thinking = thinking

    def analyze_intent(
        self, comment_body: str, parent_body: str, thread_context: str
    ) -> ReasoningResult:
        system_message = """
You are an impartial Moderation Judge for a highly contentious discussion forum.
A generated User Message is being reviewed during an active debate simulation.
Your objective is to read the Original Post, the Previous Comment, and the User
Message, then determine the true intent and severity of the message.

Evaluation rules:
1. Stance neutrality: evaluate behavior and language, not beliefs.
2. Context matters: sarcasm, idioms, quotes, and academic uses may be false positives.

Return valid JSON only with exactly:
- explanation
- category
- issue_type
- points

Categories:
- "fp" | 0 points | issue_type "benign_context"
- "flare" | 1 to 2 points | issue_type "heated_debate"
- "toxic" | 3 to 5 points | issue_type one of "personal_attack",
  "dismissive_condescending", "deadlock_loop", "weak_engagement"
- "zero-tolerance" | 100 points | issue_type "severe_explicit_hate"
"""
        user_message = f"""
Original Post:
{thread_context}

Previous Comment:
{parent_body}

User Message:
{comment_body}
"""
        return parse_reasoning_result(self._chat_json(system_message, user_message))

    def _chat_json(self, system_message: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            stream=False,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
        )
        return response_content(response).strip()


class ModerationIntervener:
    def __init__(self, client: OpenAI, model: str, thinking: bool) -> None:
        self.client = client
        self.model = model
        self.thinking = thinking

    def generate_intervention(
        self,
        text: str,
        author: str,
        cumulative_penalty: int,
        parent_text: str,
        root_context: str,
        issue_type: str,
        reasoning_explanation: str,
    ) -> InterventionPayload:
        if issue_type == "severe_explicit_hate":
            return {
                "strategy": "escalation",
                "target": "moderation_review",
                "tone_used": "neutral",
                "rationale": (
                    "The judge marked this as a severe explicit hate or "
                    "zero-tolerance case, so constructive mediation is skipped."
                ),
                "intervention_text": (
                    "Escalation required due to a severe policy violation; "
                    "constructive mediation was skipped."
                ),
            }

        system_message = """
You are a neutral discussion mediator for a Reddit-style thread.

Produce exactly one short intervention for a flagged message. The intervention
must stay neutral, avoid taking sides, avoid new factual claims, focus on tone
or reasoning quality, and be 1 to 2 sentences.

Choose exactly one strategy:
claim_refocusing, neutral_reframing, clarifying_question, evidence_reasoning,
common_ground, tone_reset, acknowledgement, hedge_soften, partial_agreement,
topic_refocus.

Return valid JSON only with exactly:
- strategy
- target
- tone_used
- rationale
- intervention_text

Set tone_used to "neutral". Target must be one of "author", "both", "thread",
or "moderation_review".
"""
        user_message = f"""
Current comment author: {author}
Current thread cumulative penalty: {cumulative_penalty}
Issue type: {issue_type}
Reasoning explanation: {reasoning_explanation}

Root context:
{root_context}

Parent text:
{parent_text}

Current comment:
{text}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            stream=False,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
        )
        return parse_intervention_payload(response_content(response), issue_type)


def fallback_intervention(issue_type: str) -> InterventionPayload:
    fallback_map: dict[str, tuple[str, str, str]] = {
        "personal_attack": (
            "tone_reset",
            "author",
            "Please drop the personal language and address the argument directly.",
        ),
        "dismissive_condescending": (
            "neutral_reframing",
            "both",
            "Please restate the disagreement without dismissive language.",
        ),
        "deadlock_loop": (
            "common_ground",
            "thread",
            "Please identify one point of agreement before continuing.",
        ),
        "weak_engagement": (
            "evidence_reasoning",
            "both",
            "Please add the reasoning or evidence behind the claim.",
        ),
    }
    strategy, target, text = fallback_map.get(
        issue_type,
        (
            "claim_refocusing",
            "author",
            "Please focus on the specific claim rather than the person making it.",
        ),
    )
    return {
        "strategy": strategy,
        "target": target,
        "tone_used": "neutral",
        "rationale": "Fallback intervention used because the model response was malformed.",
        "intervention_text": text,
    }


class ThreadStateTracker:
    def __init__(self) -> None:
        self.thread_penalty = 0
        self.cooldown_remaining = 0

    def add_penalty(self, points: int) -> int:
        self.thread_penalty += points
        return self.thread_penalty

    def reset_penalty(self) -> None:
        self.thread_penalty = 0

    def start_cooldown(self, turns: int) -> None:
        self.cooldown_remaining = max(turns, 0)

    def tick_cooldown(self) -> None:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def in_cooldown(self) -> bool:
        return self.cooldown_remaining > 0


class ActiveModerator:
    def __init__(
        self,
        reasoner: ModerationReasoner,
        intervener: ModerationIntervener,
        classifier: ToxicityClassifier | None,
        toxicity_threshold: float,
        intervention_threshold: int,
        cooldown_turns: int,
    ) -> None:
        self.reasoner = reasoner
        self.intervener = intervener
        self.classifier = classifier
        self.toxicity_threshold = toxicity_threshold
        self.intervention_threshold = intervention_threshold
        self.cooldown_turns = cooldown_turns
        self.tracker = ThreadStateTracker()

    # Adapted from ModerationOrchestrator._ingest_comment, but without the
    # NetworkX graph construction, graph node updates, or jsonl log side effects.
    def observe(
        self,
        turn: dict[str, Any],
        parent_text: str,
        thread_context: str,
    ) -> dict[str, Any] | None:
        text = str(turn.get("body", ""))
        toxicity = self.classifier.predict(text) if self.classifier else 1.0
        turn["toxicity"] = toxicity
        turn["toxicity_classifier"] = (
            str(self.classifier.model_path) if self.classifier else "skip"
        )

        if self.classifier is not None and toxicity < self.toxicity_threshold:
            return None

        reasoning = self.reasoner.analyze_intent(
            comment_body=text,
            parent_body=parent_text,
            thread_context=thread_context,
        )
        issue_type = (
            "severe_explicit_hate"
            if reasoning["category"] == "zero-tolerance"
            else reasoning["issue_type"]
        )
        is_judged_toxic = reasoning["category"] in {"toxic", "zero-tolerance"}
        cooldown_was_active = self.tracker.in_cooldown()

        thread_penalty = self.tracker.thread_penalty
        if is_judged_toxic:
            thread_penalty = self.tracker.add_penalty(reasoning["points"])

        turn["judge"] = reasoning
        turn["issue_type"] = issue_type
        turn["thread_cumulative_penalty"] = thread_penalty

        should_intervene = False
        if issue_type == "severe_explicit_hate" and is_judged_toxic:
            should_intervene = True
        elif is_judged_toxic and thread_penalty >= self.intervention_threshold:
            should_intervene = not cooldown_was_active

        if cooldown_was_active and is_judged_toxic and not should_intervene:
            self.tracker.tick_cooldown()

        if not should_intervene:
            return None

        action = self.intervener.generate_intervention(
            text=text,
            author=str(turn.get("author", "unknown")),
            cumulative_penalty=thread_penalty,
            parent_text=parent_text,
            root_context=thread_context,
            issue_type=issue_type,
            reasoning_explanation=reasoning["explanation"],
        )
        self.tracker.reset_penalty()
        self.tracker.start_cooldown(self.cooldown_turns)

        intervention_id = f"mediator_{uuid4().hex[:8]}"
        intervention = {
            "id": intervention_id,
            "author": "Active Moderator",
            "body": action["intervention_text"],
            "created_utc": time.time(),
            "replies": [],
            "round_idx": turn.get("round_idx"),
            "generated": True,
            "type": "intervention",
            "parent_id": turn.get("id"),
            "strategy": action["strategy"],
            "target": action["target"],
            "tone_used": action["tone_used"],
            "rationale": action["rationale"],
            "issue_type": issue_type,
            "points_assigned": reasoning["points"],
            "thread_cumulative_penalty": thread_penalty,
        }
        print(f"[INTERVENTION] {action['intervention_text']}\n")
        return intervention


def load_submissions(path: str | Path) -> list[RedditThread]:
    submissions: list[RedditThread] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    submissions.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
    if not submissions:
        raise ValueError(f"No submissions found in {path}")
    return submissions


def select_submission(
    submissions: list[RedditThread], submission_index: int, submission_id: str | None
) -> RedditThread:
    if submission_id is not None:
        for submission in submissions:
            if submission.get("submission_id") == submission_id:
                return submission
        raise ValueError(f"Could not find submission_id={submission_id}")

    if submission_index < 0 or submission_index >= len(submissions):
        raise ValueError(
            f"submission-index must be between 0 and {len(submissions) - 1}; "
            f"got {submission_index}"
        )
    return submissions[submission_index]


def is_usable_comment(comment: Comment, min_words: int = 1) -> bool:
    author = (comment.get("author") or "").strip()
    body = (comment.get("body") or "").strip()
    if author.lower() == "automoderator":
        return False
    if body.lower() in {"[deleted]", "[removed]"}:
        return False
    return len(body.split()) >= min_words


def find_comment_path(
    comments: list[Comment], comment_id: str, path: list[Comment] | None = None
) -> list[Comment] | None:
    path = path or []
    for comment in comments:
        next_path = path + [comment]
        if comment.get("id") == comment_id:
            return next_path
        found = find_comment_path(comment.get("replies", []), comment_id, next_path)
        if found is not None:
            return found
    return None


def iter_comment_paths(
    comments: list[Comment], min_words: int, path: list[Comment] | None = None
):
    path = path or []
    for comment in comments:
        if not is_usable_comment(comment, min_words=min_words):
            yield from iter_comment_paths(comment.get("replies", []), min_words, path)
            continue

        next_path = path + [comment]
        yield next_path
        yield from iter_comment_paths(comment.get("replies", []), min_words, next_path)


def select_seed_path(
    submission: RedditThread, comment_id: str | None, min_seed_words: int
) -> list[Comment]:
    if comment_id is not None:
        path = find_comment_path(submission.get("comments", []), comment_id)
        if path is None:
            raise ValueError(f"Could not find comment_id={comment_id}")
        return [comment for comment in path if is_usable_comment(comment)]

    paths = list(iter_comment_paths(submission.get("comments", []), min_seed_words))
    if not paths:
        raise ValueError("No usable Reddit comment chain found.")

    def score(path: list[Comment]) -> tuple[float, float, float, int]:
        toxicity_scores = [float(comment.get("toxicity", 0.0)) for comment in path]
        max_toxicity = max(toxicity_scores)
        avg_toxicity = sum(toxicity_scores) / len(toxicity_scores)
        total_toxicity = sum(toxicity_scores)
        return (max_toxicity, avg_toxicity, total_toxicity, len(path))

    return max(paths, key=score)


def select_alignment_authors(seed_path: list[Comment]) -> tuple[str, str]:
    authors: list[str] = []
    for comment in reversed(seed_path):
        if not is_usable_comment(comment):
            continue
        author = str(comment.get("author", "")).strip()
        if author not in authors:
            authors.append(author)
        if len(authors) == 2:
            return authors[1], authors[0]
    raise ValueError("Need at least two distinct usable Reddit authors in the seed chain.")


def build_alignment_profiles(
    seed_path: list[Comment],
) -> tuple[AlignmentProfile, AlignmentProfile]:
    author_1, author_2 = select_alignment_authors(seed_path)

    def profile_for(author: str, index: int) -> AlignmentProfile:
        observed_comments = [
            normalize_comment(comment)
            for comment in seed_path
            if str(comment.get("author", "")).strip() == author
        ]
        return {
            "author": author,
            "name": f"Aligned User {index} (u/{author})",
            "persona": (
                "Use only this user's observed comments in the seed chain as "
                "evidence for stance, arguments, priorities, and tone. If the "
                "evidence is limited, stay conservative and avoid inventing beliefs."
            ),
            "observed_comments": observed_comments,
        }

    return profile_for(author_1, 1), profile_for(author_2, 2)


def build_seed_history(
    submission: RedditThread, seed_path: list[Comment]
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [submission_to_comment(submission)]
    for comment in seed_path:
        history.append(normalize_comment(comment))
    return history


def submission_to_comment(submission: RedditThread) -> dict[str, Any]:
    comment: dict[str, Any] = {
        "id": str(submission.get("submission_id", "")),
        "author": str(submission.get("author", "unknown")),
        "body": format_submission_text(submission),
        "created_utc": float(submission.get("created_utc", 0.0)),
        "replies": [],
        "type": "post",
        "generated": False,
    }
    if "body_toxicity" in submission:
        comment["toxicity"] = float(submission["body_toxicity"])
    return comment


def normalize_comment(comment: Comment) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "id": str(comment.get("id", "")),
        "author": str(comment.get("author", "unknown")),
        "body": (comment.get("body") or "").strip(),
        "created_utc": float(comment.get("created_utc", 0.0)),
        "replies": [],
        "type": "comment",
        "generated": False,
    }
    if "toxicity" in comment:
        normalized["toxicity"] = float(comment["toxicity"])
    return normalized


def format_submission_text(submission: RedditThread) -> str:
    title = (submission.get("title") or "").strip()
    selftext = (submission.get("selftext") or "").strip()
    if selftext:
        return f"{title}\n\n{selftext}"
    return title


def thread_context_for(submission: RedditThread) -> str:
    post_body = (submission.get("selftext") or "").strip()
    if post_body:
        return f"Title: {submission['title']}\nPost: {post_body}"
    return f"Title: {submission['title']}"


def print_seed_context(submission: RedditThread, seed_path: list[Comment]) -> None:
    print("=" * 50)
    print(f"Selected Reddit submission: {submission.get('title')}")
    print(f"Submission ID: {submission.get('submission_id')}")
    print(f"Seed comment chain length: {len(seed_path)}")
    print("=" * 50)
    for idx, comment in enumerate(seed_path, start=1):
        body = " ".join((comment.get("body") or "").split())
        preview = body[:240] + ("..." if len(body) > 240 else "")
        print(f"{idx}. u/{comment.get('author')} ({comment.get('id')}): {preview}")
    print("=" * 50)


def print_alignment_profiles(
    profiles: tuple[AlignmentProfile, AlignmentProfile]
) -> None:
    print("Aligned Reddit users:")
    for idx, profile in enumerate(profiles, start=1):
        print(
            f"Agent {idx}: u/{profile['author']} "
            f"({len(profile['observed_comments'])} observed comments)"
        )
    print("=" * 50)


def reddit_thread_debate_with_moderation(
    agents: tuple[RedditDebateAgent, RedditDebateAgent],
    moderator: ActiveModerator,
    submission: RedditThread,
    history: list[dict[str, Any]],
    topic: str,
    debate_round: int,
    first_agent: str,
    out_dir: str,
    model: str,
    toxicity_classifier: str,
    judge_model: str,
    intervention_model: str,
) -> list[dict[str, Any]]:
    debate_id = uuid4().hex[:5]
    agent_order = agents if first_agent == "1" else (agents[1], agents[0])
    root_context = thread_context_for(submission)

    for round_idx in range(1, debate_round + 1):
        print(f"Generated Round {round_idx} (Debate ID: {debate_id})")
        print("-" * 50)
        for agent in agent_order:
            parent_text = str(history[-1].get("body", "")) if history else ""
            text = agent.speak(history=history, submission=submission)
            generated_turn = {
                "id": f"generated_{uuid4().hex[:8]}",
                "author": agent.name,
                "body": text,
                "created_utc": time.time(),
                "replies": [],
                "round_idx": round_idx,
                "generated": True,
                "type": "comment",
            }
            history.append(generated_turn)

            intervention = moderator.observe(
                turn=generated_turn,
                parent_text=parent_text,
                thread_context=root_context,
            )
            if intervention is not None:
                history.append(intervention)
            print("-" * 50)

    save_history(
        debate_id=debate_id,
        history=history,
        submission=submission,
        topic=topic,
        out_dir=out_dir,
        model=model,
        toxicity_classifier=toxicity_classifier,
        judge_model=judge_model,
        intervention_model=intervention_model,
    )
    return history


def save_history(
    debate_id: str,
    history: list[dict[str, Any]],
    submission: RedditThread,
    topic: str,
    out_dir: str,
    model: str | None = None,
    toxicity_classifier: str | None = None,
    judge_model: str | None = None,
    intervention_model: str | None = None,
) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_suffix = f"_{safe_filename_piece(model)}" if model else ""
    topic_name = safe_filename_piece(topic)
    submission_id = safe_filename_piece(
        str(submission.get("submission_id", "unknown_submission"))
    )
    out_path = (
        Path(out_dir)
        / f"moderated_reddit_{topic_name}_{submission_id}_{debate_id}{model_suffix}.jsonl"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for turn_idx, turn in enumerate(history):
            record = {
                "debate_id": debate_id,
                "submission_id": submission.get("submission_id"),
                "submission_url": submission.get("submission_url"),
                "title": submission.get("title"),
                "toxicity_classifier": toxicity_classifier,
                "judge_model": judge_model,
                "intervention_model": intervention_model,
                "turn_idx": turn_idx,
                "round_idx": turn.get("round_idx"),
                **turn,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Moderated Reddit debate history saved to {out_path}")
    return out_path


def safe_filename_piece(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue a real Reddit abortion debate thread with aligned users "
            "and an active moderation agent. This script is standalone and does "
            "not use the NetworkX moderation graph."
        )
    )

    # Input and output files
    parser.add_argument("--input", default="out.jsonl", help="Path to Reddit jsonl data.")
    parser.add_argument("--out-dir", default="sim_debate_records", help="Output directory.")

    # Debate model settings. These use Ollama's OpenAI-compatible local endpoint.
    parser.add_argument("--base-url", default="http://localhost:11434/v1/")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--model", default="llama3.1:8b", help="Debate model.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    # Ollama's OpenAI-compatible API may ignore --no-thinking for some models.
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True, help="Request model thinking mode via chat_template_kwargs.")

    # Active moderation model settings. These reuse the same OpenAI-compatible endpoint.
    parser.add_argument("--judge-model", default="llama3.2:3b")
    parser.add_argument("--intervention-model", default=None)

    # Active moderation policy settings.
    parser.add_argument(
        "--classifier",
        default="models/cga_deberta_onnx_int8",
        help="Local CGA classifier path, or 'skip' to judge every generated turn.",
    )
    parser.add_argument("--toxicity-threshold", type=float, default=0.6)
    parser.add_argument("--intervention-threshold", type=int, default=10)
    parser.add_argument("--cooldown-turns", type=int, default=2)

    # Reddit seed selection settings.
    parser.add_argument("--submission-index", type=int, default=0)
    parser.add_argument("--submission-id", default=None)
    parser.add_argument("--comment-id", default=None)
    parser.add_argument("--min-seed-words", type=int, default=8)

    # Debate generation settings.
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--first-agent", choices=["1", "2"], default="1")
    parser.add_argument("--max-context-turns", type=int, default=10)
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Only load, select, print, and save the real Reddit seed thread.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topic = "abortion rights"

    submissions = load_submissions(args.input)
    submission = select_submission(
        submissions, args.submission_index, args.submission_id
    )
    seed_path = select_seed_path(submission, args.comment_id, args.min_seed_words)
    history = build_seed_history(submission, seed_path)
    alignment_profiles = build_alignment_profiles(seed_path)

    print_seed_context(submission, seed_path)
    print_alignment_profiles(alignment_profiles)
    print(f"Topic: {topic}")
    print(f"Debate Model: {args.model}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Intervention Model: {args.intervention_model or args.judge_model}")
    print(f"Toxicity Classifier: {args.classifier}")
    print(f"Generated Rounds: {args.rounds}")
    print(f"Thinking requested: {args.thinking}")
    print("=" * 50)

    if args.no_generate:
        save_history(
            debate_id=uuid4().hex[:5],
            history=history,
            submission=submission,
            topic=topic,
            out_dir=args.out_dir,
            model=args.model,
            toxicity_classifier=args.classifier,
            judge_model=args.judge_model,
            intervention_model=args.intervention_model or args.judge_model,
        )
        return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    classifier = (
        None if args.classifier == "skip" else ToxicityClassifier(args.classifier)
    )
    moderator = ActiveModerator(
        reasoner=ModerationReasoner(client, args.judge_model, args.thinking),
        intervener=ModerationIntervener(
            client, args.intervention_model or args.judge_model, args.thinking
        ),
        classifier=classifier,
        toxicity_threshold=args.toxicity_threshold,
        intervention_threshold=args.intervention_threshold,
        cooldown_turns=args.cooldown_turns,
    )

    agent_1 = RedditDebateAgent(
        client=client,
        model=args.model,
        stream=args.stream,
        thinking=args.thinking,
        topic=topic,
        name=alignment_profiles[0]["name"],
        persona=alignment_profiles[0]["persona"],
        aligned_author=alignment_profiles[0]["author"],
        observed_comments=alignment_profiles[0]["observed_comments"],
        max_context_turns=args.max_context_turns,
    )
    agent_2 = RedditDebateAgent(
        client=client,
        model=args.model,
        stream=args.stream,
        thinking=args.thinking,
        topic=topic,
        name=alignment_profiles[1]["name"],
        persona=alignment_profiles[1]["persona"],
        aligned_author=alignment_profiles[1]["author"],
        observed_comments=alignment_profiles[1]["observed_comments"],
        max_context_turns=args.max_context_turns,
    )

    reddit_thread_debate_with_moderation(
        agents=(agent_1, agent_2),
        moderator=moderator,
        submission=submission,
        history=history,
        topic=topic,
        debate_round=args.rounds,
        first_agent=args.first_agent,
        out_dir=args.out_dir,
        model=args.model,
        toxicity_classifier=args.classifier,
        judge_model=args.judge_model,
        intervention_model=args.intervention_model or args.judge_model,
    )


if __name__ == "__main__":
    main()
