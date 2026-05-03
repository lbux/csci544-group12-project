# pyright: reportExplicitAny=false, reportAny=false, reportUnknownMemberType=false, reportAssignmentType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from huggingface_hub import snapshot_download
from openai import OpenAI
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from .llm_client import generate_structured_output
from .schemas import InterventionResult, ReasoningResult


class ToxicityClassifier:
    """Uses a fine tuning deberta classifier to classify toxicity for the current message"""
    def __init__(self, model_path: str | Path = "models/cga_deberta_onnx_int8") -> None:
        self.model_path: Path = Path(model_path)
        if not self.model_path.exists():
            snapshot_download(
                repo_id="lbux/cga-deberta-onnx-int8", local_dir=self.model_path
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.ort_model = ORTModelForSequenceClassification.from_pretrained(
            self.model_path, file_name="model_quantized.onnx"
        )

    def predict(self, text: str) -> float:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.ort_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)  # pyright: ignore[reportPrivateImportUsage]
        return float(probs[0][1].item())


class ModerationReasoner:
    def __init__(self, client: OpenAI, model: str, thinking: bool) -> None:
        self.client: OpenAI = client
        self.model: str = model
        self.thinking: bool = thinking

    def analyze_intent(
        self, comment_body: str, parent_body: str, thread_context: str
    ) -> ReasoningResult:
        system_message = "You are an impartial Moderation Judge for a contentious forum. Evaluate the *behavior and language*, not beliefs. Return valid JSON only. Categories: 'fp' (0 pts, benign_context), 'flare' (1-2 pts, heated_debate), 'toxic' (3-5 pts, personal_attack, dismissive_condescending, deadlock_loop, weak_engagement), 'zero-tolerance' (100 pts, severe_explicit_hate)."
        user_message = f"Original Post:\n{thread_context}\n\nPrevious Comment:\n{parent_body}\n\nUser Message:\n{comment_body}"

        try:
            return generate_structured_output(
                self.client,
                self.model,
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                ReasoningResult,
                self.thinking,
            )
        except Exception:
            return ReasoningResult(
                explanation="Fallback classification due to malformed output.",
                category="toxic",
                issue_type="unclassified",
                points=3,
            )


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
        root_context: str,  # pyright: ignore[reportUnusedParameter]
        issue_type: str,
        reasoning_explanation: str,
    ) -> InterventionResult:
        if issue_type == "severe_explicit_hate":
            return InterventionResult(
                strategy="escalation",
                target="moderation_review",
                tone_used="neutral",
                rationale="Severe policy violation.",
                intervention_text="Escalation required due to severe violation.",
            )

        system_message = "You are a neutral discussion mediator. Produce one short intervention (1-2 sentences). Strategy choices: claim_refocusing, neutral_reframing, clarifying_question, evidence_reasoning, common_ground, tone_reset, acknowledgement, hedge_soften, partial_agreement, topic_refocus. Return valid JSON only."
        user_message = f"Author: {author}\nPenalty: {cumulative_penalty}\nIssue: {issue_type}\nReason: {reasoning_explanation}\n\nParent: {parent_text}\n\nComment: {text}"

        try:
            return generate_structured_output(
                self.client,
                self.model,
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                InterventionResult,
                self.thinking,
            )
        except Exception:
            return InterventionResult(
                strategy="tone_reset",
                target="author",
                tone_used="neutral",
                rationale="Fallback intervention.",
                intervention_text="Please restate the disagreement calmly without personal attacks.",
            )


class ThreadStateTracker:
    def __init__(self) -> None:
        self.thread_penalty: int = 0
        self.cooldown_remaining: int = 0

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
        toxicity_threshold: float = 0.75,
        intervention_threshold: int = 10,
        cooldown_turns: int = 2,
    ):
        self.reasoner = reasoner
        self.intervener = intervener
        self.classifier = classifier
        self.toxicity_threshold = toxicity_threshold
        self.intervention_threshold = intervention_threshold
        self.cooldown_turns = cooldown_turns
        self.tracker = ThreadStateTracker()

    def observe(
        self, turn: dict[str, Any], parent_text: str, thread_context: str
    ) -> dict[str, Any] | None:
        text = str(turn.get("body", ""))
        toxicity: float = (
            self.classifier.predict(text) if self.classifier is not None else 1.0
        )
        turn["toxicity"] = toxicity

        if self.classifier is not None and toxicity < self.toxicity_threshold:
            return None

        reasoning = self.reasoner.analyze_intent(text, parent_text, thread_context)
        issue_type = (
            "severe_explicit_hate"
            if reasoning.category == "zero-tolerance"
            else reasoning.issue_type
        )
        is_judged_toxic = reasoning.category in {"toxic", "zero-tolerance"}
        cooldown_active = self.tracker.in_cooldown()

        thread_penalty = (
            self.tracker.add_penalty(reasoning.points)
            if is_judged_toxic
            else self.tracker.thread_penalty
        )

        turn["judge"] = reasoning.model_dump()
        turn["issue_type"] = issue_type
        turn["thread_cumulative_penalty"] = thread_penalty

        should_intervene = False
        if issue_type == "severe_explicit_hate" and is_judged_toxic:
            should_intervene = True
        elif (
            is_judged_toxic
            and thread_penalty >= self.intervention_threshold
            and not cooldown_active
        ):
            should_intervene = True

        if cooldown_active and is_judged_toxic and not should_intervene:
            self.tracker.tick_cooldown()

        if not should_intervene:
            return None

        author_str = str(turn.get("author", "unknown"))
        action = self.intervener.generate_intervention(
            text,
            author_str,
            thread_penalty,
            parent_text,
            thread_context,
            issue_type,
            reasoning.explanation,
        )

        self.tracker.reset_penalty()
        self.tracker.start_cooldown(self.cooldown_turns)

        print(f"[INTERVENTION] {action.intervention_text}\n")

        return {
            "id": f"mediator_{uuid4().hex[:8]}",
            "author": "Active Moderator",
            "body": action.intervention_text,
            "created_utc": time.time(),
            "replies": [],
            "round_idx": turn.get("round_idx"),
            "generated": True,
            "type": "intervention",
            "parent_id": turn.get("id"),
            **action.model_dump(),
        }
