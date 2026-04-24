import argparse
import json
import time
from pathlib import Path
from typing import NotRequired, TypedDict
from uuid import uuid4

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


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


class RedditDebateAgent:
    """
    RedditDebateAgent continues a real Reddit discussion thread.

    Unlike abortion_demo.py, the initial context is not empty. The debate starts
    from a real Reddit submission/comment chain loaded from out.jsonl, then the
    LLM writes the next nested reply in the thread.
    """

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
    ):
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

    def speak(self, history: list[Comment], submission: RedditThread) -> str:
        chat_messages = self._build_chat_messages(history, submission)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            stream=self.stream,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
        )

        if self.stream is False:
            text = response.choices[0].message.content.strip()
            print(f"{self.name} replies: {text}\n")
        else:
            text = self._stream_response(response)

        return text

    def _build_chat_messages(self, history: list[Comment], submission: RedditThread) -> list[ChatCompletionMessageParam]:
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
                    f"You are {self.name}, an AI debate participant aligned with the observed Reddit user u/{self.aligned_author}. "
                    f"You are participating in a Reddit thread debate about {self.topic}. "
                    f"Your alignment goal: {self.persona} "
                    "Do not claim to be the real Reddit user. "
                    "Write as a simulated Reddit participant whose stance, arguments, and tone stay consistent with the observed user."
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

    def _stream_response(self, response) -> str:
        parts = []
        print(f"{self.name}: ", end="", flush=True)
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                parts.append(delta.content)
        print()

        return "".join(parts).strip()



def load_submissions(path: str | Path) -> list[RedditThread]:
    submissions: list[RedditThread] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    submissions.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
    if not submissions:
        raise ValueError(f"No submissions found in {path}")

    return submissions


def select_submission(submissions: list[RedditThread], submission_index: int, submission_id: str | None) -> RedditThread:
    if submission_id is not None:
        for submission in submissions:
            if submission.get("submission_id") == submission_id:

                return submission
        raise ValueError(f"Could not find submission_id={submission_id}")

    if submission_index < 0 or submission_index >= len(submissions):
        raise ValueError(
            f"submission-index must be between 0 and {len(submissions) - 1}; got {submission_index}"
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


def find_comment_path(comments: list[Comment], comment_id: str, path: list[Comment] | None = None) -> list[Comment] | None:
    path = path or []
    for comment in comments:
        next_path = path + [comment]
        if comment.get("id") == comment_id:

            return next_path
        found = find_comment_path(comment.get("replies", []), comment_id, next_path)
        if found is not None:

            return found

    return None


def iter_comment_paths(comments: list[Comment], min_words: int, path: list[Comment] | None = None):
    path = path or []
    for comment in comments:
        if not is_usable_comment(comment, min_words=min_words):
            yield from iter_comment_paths(comment.get("replies", []), min_words, path)
            continue

        next_path = path + [comment]
        yield next_path
        yield from iter_comment_paths(comment.get("replies", []), min_words, next_path)


def select_seed_path(submission: RedditThread, comment_id: str | None, min_seed_words: int) -> list[Comment]:
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

    raise ValueError("Need at least two distinct usable Reddit authors in the seed chain for alignment.")


def build_alignment_profiles(seed_path: list[Comment]) -> tuple[AlignmentProfile, AlignmentProfile]:
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
                "Use only this user's observed comments in the seed chain as evidence for stance, "
                "arguments, priorities, and tone. If the evidence is limited, stay conservative and avoid inventing beliefs."
            ),
            "observed_comments": observed_comments,
        }

    return profile_for(author_1, 1), profile_for(author_2, 2)


def print_alignment_profiles(profiles: tuple[AlignmentProfile, AlignmentProfile]) -> None:
    print("Aligned Reddit users:")
    for idx, profile in enumerate(profiles, start=1):
        print(f"Agent {idx}: u/{profile['author']} ({len(profile['observed_comments'])} observed comments)")
    print("=" * 50)


def build_seed_history(submission: RedditThread, seed_path: list[Comment]) -> list[Comment]:
    history: list[Comment] = [submission_to_comment(submission)]
    for comment in seed_path:
        history.append(normalize_comment(comment))

    return history


def submission_to_comment(submission: RedditThread) -> Comment:
    comment: Comment = {
        "id": str(submission.get("submission_id", "")),
        "author": str(submission.get("author", "unknown")),
        "body": format_submission_text(submission),
        "created_utc": float(submission.get("created_utc", 0.0)),
        "replies": [],
    }
    if "body_toxicity" in submission:
        comment["toxicity"] = float(submission["body_toxicity"])

    return comment


def normalize_comment(comment: Comment) -> Comment:
    normalized: Comment = {
        "id": str(comment.get("id", "")),
        "author": str(comment.get("author", "unknown")),
        "body": (comment.get("body") or "").strip(),
        "created_utc": float(comment.get("created_utc", 0.0)),
        "replies": [],
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


def reddit_thread_debate(
    agents: tuple[RedditDebateAgent, RedditDebateAgent],
    submission: RedditThread,
    history: list[Comment],
    topic: str,
    debate_round: int,
    first_agent: str,
    out_dir: str = "sim_debate_records",
    model: str | None = None,
) -> list[Comment]:
    debate_id = uuid4().hex[:5]
    agent_order = agents if first_agent == "1" else (agents[1], agents[0])

    for round_idx in range(1, debate_round + 1):
        print(f"Generated Round {round_idx} (Debate ID: {debate_id})")
        print("-" * 50)
        for agent in agent_order:
            text = agent.speak(history=history, submission=submission)
            history.append(
                {
                    "id": f"generated_{uuid4().hex[:8]}",
                    "author": agent.name,
                    "body": text,
                    "created_utc": time.time(),
                    "replies": [],
                    "round_idx": round_idx,
                }
            )
            print("-" * 50)

    save_history(debate_id, history, submission, topic, out_dir, model)

    return history


def save_history(
    debate_id: str,
    history: list[Comment],
    submission: RedditThread,
    topic: str,
    out_dir: str = "sim_debate_records",
    model: str | None = None,
) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_suffix = f"_{safe_filename_piece(model)}" if model else ""
    topic_name = safe_filename_piece(topic)
    submission_id = safe_filename_piece(str(submission.get("submission_id", "unknown_submission")))
    out_path = Path(out_dir) / f"reddit_{topic_name}_{submission_id}_{debate_id}{model_suffix}.jsonl"

    with open(out_path, "w") as f:
        for turn_idx, turn in enumerate(history):
            record = {
                "debate_id": debate_id,
                "submission_id": submission.get("submission_id"),
                "submission_url": submission.get("submission_url"),
                "title": submission.get("title"),
                "turn_idx": turn_idx,
                "round_idx": turn.get("round_idx"),
                "generated": turn["id"].startswith("generated_"),
                **turn,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Reddit debate history saved to {out_path}")

    return out_path


def safe_filename_piece(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Continue a real Reddit abortion debate thread with LLM agents."
    )

    # input and output settings
    parser.add_argument("--input", default="out.jsonl", help="Path to Reddit jsonl data.")
    parser.add_argument("--out-dir", default="sim_debate_records", help="Output directory.")

    # API and model settings
    parser.add_argument("--base-url", default="http://localhost:11434/v1/")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--model", default="llama3.1:8b", help="Model to use.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    # Ollama's OpenAI-compatible API may ignore --no-thinking for some models.
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True, help="Request model thinking mode via chat_template_kwargs.")

    # Reddit seed selection settings
    parser.add_argument("--submission-index", type=int, default=0)
    parser.add_argument("--submission-id", default=None)
    parser.add_argument("--comment-id", default=None, help="Use the path ending at this comment.")
    parser.add_argument("--min-seed-words", type=int, default=8, help="Minimum words for comments used when auto-selecting a seed chain.")

    # Debate settings
    parser.add_argument("--rounds", type=int, default=3, help="Rounds of generated replies.")
    parser.add_argument("--first-agent", choices=["1", "2"], default="1")
    parser.add_argument("--max-context-turns", type=int, default=10)
    parser.add_argument("--no-generate", action="store_true", help="Only load, select, print, and save the real Reddit seed thread.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = args.model
    stream = args.stream
    thinking = args.thinking
    debate_round = args.rounds
    TOPIC = "abortion rights"

    submissions = load_submissions(args.input)
    submission = select_submission(submissions, args.submission_index, args.submission_id)
    seed_path = select_seed_path(submission, args.comment_id, args.min_seed_words)
    history = build_seed_history(submission, seed_path)
    alignment_profiles = build_alignment_profiles(seed_path)

    print_seed_context(submission, seed_path)
    print_alignment_profiles(alignment_profiles)
    print(f"Topic: {TOPIC}")
    print(f"LLM Model: {model}")
    print(f"Generated Rounds: {debate_round}")
    print(f"Stream: {stream}")
    print(f"Thinking requested: {thinking}")
    print("=" * 50)

    if args.no_generate:
        save_history(uuid4().hex[:5], history, submission, TOPIC, args.out_dir, model)
    else:
        client = OpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
        )

        agent_1 = RedditDebateAgent(
            client=client,
            model=model,
            stream=stream,
            thinking=thinking,
            topic=TOPIC,
            name=alignment_profiles[0]["name"],
            persona=alignment_profiles[0]["persona"],
            aligned_author=alignment_profiles[0]["author"],
            observed_comments=alignment_profiles[0]["observed_comments"],
            max_context_turns=args.max_context_turns,
        )
        agent_2 = RedditDebateAgent(
            client=client,
            model=model,
            stream=stream,
            thinking=thinking,
            topic=TOPIC,
            name=alignment_profiles[1]["name"],
            persona=alignment_profiles[1]["persona"],
            aligned_author=alignment_profiles[1]["author"],
            observed_comments=alignment_profiles[1]["observed_comments"],
            max_context_turns=args.max_context_turns,
        )

        reddit_thread_debate(
            agents=(agent_1, agent_2),
            submission=submission,
            history=history,
            topic=TOPIC,
            debate_round=debate_round,
            first_agent=args.first_agent,
            out_dir=args.out_dir,
            model=model,
        )
