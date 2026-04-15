import json
from pathlib import Path
from typing import Any, cast

import ollama
import torch
from huggingface_hub import (
    snapshot_download,  # pyright: ignore[reportUnknownVariableType]
)
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer

from interfaces import InterventionResult, ReasoningResult


class _InterventionPayload(BaseModel):
    strategy: str
    target: str
    tone_used: str
    rationale: str
    intervention_text: str


def _extract_json_dict(raw_text: str) -> dict[str, Any] | None:
    candidates = [raw_text.strip()]
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw_text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class LLMReasoner:
    def __init__(self, model: str = "gemma4:e4b") -> None:
        self.model: str = model
        # self.client: Client = ollama.Client(
        #     host="http://localhost:14434",
        # )
        _ = ollama.pull(self.model)

    def analyze_intent(
        self, comment_body: str, parent_body: str, thread_context: str
    ) -> ReasoningResult:

        SYSTEM_MESSAGE: str = """
        You are an impartial Moderation Judge for a highly contentious discussion forum. 
        An automated toxicity classifier has flagged a User Message. Your objective is to read the Original Post (the topic), the Previous Comment (the immediate context), and the flagged User Message, and determine the true intent and severity of the message.

        EVALUATION RULES:
        1. STANCE NEUTRALITY: You must evaluate the *behavior and language* of the user, not their beliefs. Both sides of an argument must be held to the exact same behavioral standard.
        2. CONTEXT MATTERS: Sarcasm, idioms, and hyperbolic jokes might trigger the automated classifier but are not actual violations. 

        RUBRIC & CATEGORIES:
        You must strictly assign one of the following categories and an associated point value:

        - "fp" (False Positive) | 0 points
        The classifier made a mistake. The text contains flagged words used in a benign, joking, or purely academic context.
        issue_type should be "benign_context"

        - "flare" (Heated Debate) | 1 to 2 points
        The comment is harsh, highly sarcastic, or deeply critical of the *argument*. It has rough edges typical of contentious debates, but it attacks the *ideas*, not the *person*. It does not cross the line into abuse.
        issue_type should be "heated_debate"

        - "toxic" (Ad Hominem/Abuse) | 3 to 5 points
        The comment attacks the *person* rather than the argument. It includes insults, severe condescension, harassment, or aggressive attempts to shut down the conversation. It is destructive to healthy debate.
        issue_type must be one of:
        - "personal_attack"
        - "dismissive_condescending"
        - "deadlock_loop"
        - "weak_engagement"

        - "zero-tolerance" (Severe Violation) | 100 points
        The comment contains hate speech, extreme slurs, threats of real-world violence, doxing, or encouragement of self-harm. Warrants an immediate ban.
        issue_type should be "severe_explicit_hate"

        FEW-SHOT EXAMPLES:

        Example 1:
        Context: Post about dietary preferences. Previous: "I love pineapple on pizza."
        User Message: "You are literally killing me with that opinion lol, I'm dying."
        Output:
        {
        "explanation": "The user used the words 'killing' and 'dying', which likely triggered the automated classifier. However, within the context of a pizza debate, this is clearly benign hyperbole and a joke.",
        "category": "fp",
        "issue_type": "benign_context",
        "points": 0
        }

        Example 2:
        Context: Post about economic policy. Previous: "We need to print more money to solve this."
        User Message: "That is completely absurd. Your data is 10 years out of date and frankly, it's a ridiculous and financially illiterate argument."
        Output:
        {
        "explanation": "The user is being highly combative and harsh. However, words like 'absurd', 'ridiculous', and 'financially illiterate' are directed entirely at the argument and the data, not attacking the user personally. This is acceptable heated debate.",
        "category": "flare",
        "issue_type": "heated_debate",
        "points": 1
        }

        Example 3:
        Context: Post about local elections. Previous: "I think candidate A has better infrastructure plans."
        User Message: "You are a braindead moron. He will bankrupt us"
        Output:
        {
        "explanation": "The user is deploying direct ad hominem insults ('braindead moron'). This crosses the line from debating ideas to personal abuse.",
        "category": "toxic",
        "issue_type": "personal_attack",
        "points": 5
        }

        TASK:
        Analyze the following inputs and output a valid JSON object matching the requested schema. Provide the explanation, then category, then issue_type, then points.
        """

        USER_MESSAGE: str = f"""TASK:
        Analyze the following inputs and output a valid JSON object matching the requested schema. Provide the explanation, then category, then issue_type, then points.

        Original Post:
        {parent_body}

        Previous Comment:
        {thread_context}

        User Message:
        {comment_body}
        """

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE},
        ]

        response: ollama.ChatResponse = ollama.chat(  # pyright: ignore[reportUnknownMemberType]
            self.model, messages, format=ReasoningResult.model_json_schema()
        )
        try:
            return ReasoningResult.model_validate_json(response.message.content)  # pyright: ignore[reportArgumentType]
        except ValidationError:
            parsed = _extract_json_dict(response.message.content)
            if parsed is not None:
                parsed.setdefault("issue_type", "unclassified")
                try:
                    return ReasoningResult.model_validate(parsed)
                except ValidationError:
                    pass

        return ReasoningResult(
            explanation="Model output was malformed; applied a conservative fallback classification for a flagged comment.",
            category="toxic",
            issue_type="unclassified",
            points=3,
        )


class LLMIntervener:
    def __init__(self, model: str = "gemma4:e4b") -> None:
        self.model: str = model
        _ = ollama.pull(self.model)

    def generate_intervention(
        self,
        text: str,
        author: str,
        cumulative_penalty: int,
        parent_text: str,
        root_context: str,
        issue_type: str,
        reasoning_explanation: str,
    ) -> InterventionResult:
        if issue_type == "severe_explicit_hate":
            return {
                "strategy": "escalation",
                "target": "moderation_review",
                "tone_used": "neutral",
                "rationale": "The judge marked this as a severe explicit hate or zero-tolerance case, so constructive mediation is skipped in favor of escalation.",
                "intervention_text": "Escalation required due to a severe policy violation; constructive mediation was skipped.",
            }

        SYSTEM_MESSAGE = """
        You are a neutral discussion mediator for a Reddit-style thread.

        Your job is to produce exactly one short intervention for a flagged message in a thread-level moderation system.
        The intervention must:
        - stay neutral
        - avoid taking sides
        - avoid adding new factual claims
        - focus on tone, reasoning quality, or engagement quality
        - be 1 to 2 sentences
        - choose exactly one strategy from this inventory:
          1. claim_refocusing
          2. neutral_reframing
          3. clarifying_question
          4. evidence_reasoning
          5. common_ground
          6. tone_reset
          7. acknowledgement
          8. hedge_soften
          9. partial_agreement
          10. topic_refocus

        Strategy guidance:
        - personal_attack -> tone_reset or claim_refocusing; target can be "author" or "both"
        - dismissive_condescending -> neutral_reframing or acknowledgement; target can be "author" or "both"
        - deadlock_loop -> common_ground or topic_refocus; prefer target "both" or "thread"
        - weak_engagement -> evidence_reasoning or clarifying_question; prefer target "both" or "thread"

        Return valid JSON only with:
        - strategy
        - target
        - tone_used
        - rationale
        - intervention_text

        For v1, set tone_used to "neutral".
        Because this is thread-level moderation, choose the most appropriate target from "author", "both", "thread", or "moderation_review".
        """

        USER_MESSAGE = f"""Current comment author: {author}
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

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE},
        ]

        response: ollama.ChatResponse = ollama.chat(  # pyright: ignore[reportUnknownMemberType]
            self.model, messages, format=_InterventionPayload.model_json_schema()
        )

        return self._parse_intervention_response(
            raw_content=response.message.content,
            issue_type=issue_type,
        )

    def _parse_intervention_response(
        self, raw_content: str, issue_type: str
    ) -> InterventionResult:
        try:
            payload = _InterventionPayload.model_validate_json(raw_content)
            return cast(InterventionResult, payload.model_dump())
        except ValidationError:
            parsed = _extract_json_dict(raw_content)
            if parsed is not None:
                try:
                    payload = _InterventionPayload.model_validate(parsed)
                    return cast(InterventionResult, payload.model_dump())
                except ValidationError:
                    pass

        return self._fallback_intervention(issue_type)

    def _fallback_intervention(self, issue_type: str) -> InterventionResult:
        strategy_map: dict[str, tuple[str, str, str]] = {
            "personal_attack": (
                "tone_reset",
                "author",
                "The comment appears to target the person instead of the claim, so the intervention resets tone and redirects attention to the argument.",
            ),
            "dismissive_condescending": (
                "neutral_reframing",
                "both",
                "The comment reads as dismissive, so the intervention reframes the exchange in a neutral way without endorsing either side.",
            ),
            "deadlock_loop": (
                "common_ground",
                "thread",
                "The exchange appears stuck in a repeat loop, so the intervention tries to restart progress by identifying shared ground.",
            ),
            "weak_engagement": (
                "evidence_reasoning",
                "both",
                "The thread is light on supporting reasoning, so the intervention asks for clearer evidence or explanation from the discussion.",
            ),
        }
        strategy, target, rationale = strategy_map.get(
            issue_type,
            (
                "claim_refocusing",
                "author",
                "The fallback intervention redirects the discussion toward the specific claim because the model response was malformed.",
            ),
        )

        messages: dict[str, str] = {
            "claim_refocusing": "Please focus on the specific claim rather than the person making it. Explain which part of the argument you disagree with and why.",
            "neutral_reframing": "Please restate the disagreement without dismissive language. A neutral explanation will make the point easier to engage with.",
            "clarifying_question": "Could you clarify the specific claim you are challenging? A concrete question will help keep the discussion focused.",
            "evidence_reasoning": "Please add the reasoning or evidence behind the claim. That will make it easier to respond to the substance of the point.",
            "common_ground": "Please identify one point of agreement before continuing. That can help move the discussion out of a repeat loop.",
            "tone_reset": "Please drop the personal language and address the argument directly. A calmer restatement will make the disagreement easier to engage with.",
            "acknowledgement": "Please acknowledge the other point before disagreeing. That can lower the temperature without changing your position.",
            "hedge_soften": "Please soften the certainty of the claim and explain the reasoning. That makes disagreement easier to discuss productively.",
            "partial_agreement": "Please note any part of the other comment you agree with before explaining the disagreement. That can keep the exchange substantive.",
            "topic_refocus": "Please return to the main claim under discussion and avoid repeating the same back-and-forth. A focused response will be more useful here.",
        }

        return {
            "strategy": strategy,
            "target": target,
            "tone_used": "neutral",
            "rationale": rationale,
            "intervention_text": messages[strategy],
        }


class ToxicityClassifier:
    def __init__(self) -> None:
        model_path = Path("models/cga_deberta_onnx_int8")

        if not model_path.exists():
            _ = snapshot_download(
                repo_id="lbux/cga-deberta-onnx-int8",
                local_dir=model_path,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownMemberType, reportUnannotatedClassAttribute]

        self.ort_model = ORTModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnannotatedClassAttribute]
            model_path,
            file_name="model_quantized.onnx",
        )

    def predict(self, text: str) -> float:

        # print(self.ort_model.config)

        inputs = self.tokenizer(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            text, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = self.ort_model(**inputs)  # pyright: ignore[reportUnknownVariableType]

        logits = outputs.logits  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        probs = torch.softmax(input=logits, dim=-1)  # pyright: ignore[reportUnknownArgumentType]
        toxicity_score = probs[0][1].item()

        return toxicity_score


# if __name__ == "__main__":
#     classifier = ToxicityClassifier()
#     reasoner = LLMReasoner()
