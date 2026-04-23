"""Debate-aware LLM-as-judge with a rubric tuned to catch subtle hostility
patterns that are common in online debate but miss the original rubric's
explicit-attack focus. Leaves models.LLMReasoner untouched so the
simulation orchestrator's behavior does not change.

Subtle-toxic patterns promoted into `toxic` category:
  - condescension ("let me try to explain this more simply")
  - mocking sarcasm ("I love how eager you are to...")
  - bad-faith accusations ("you're still dodging", "moving the goalposts")
  - strawmanning (visible misrepresentation of opponent)
  - dismissive framing ("this isn't worth engaging with")
  - deadlock behavior (repeating without engaging counter-points)

Rubric categories and point values are otherwise identical to the
original LLMReasoner so that downstream accounting (threshold, cooldown,
intervention_logs) remains drop-in compatible.
"""

from __future__ import annotations

import json
from typing import Any

import ollama
from pydantic import ValidationError

from interfaces import ReasoningResult


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


DEBATE_SYSTEM_MESSAGE = """
You are an impartial Moderation Judge for an online *debate* forum. An automated toxicity classifier has flagged a User Message, or a pipeline has routed every turn to you for analysis. Your objective is to read the Original Post (topic), the Previous Comment (immediate context), and the flagged User Message, and determine the true intent and severity of the message, with special attention to **subtle hostility patterns that are common in debate but easy to miss**.

EVALUATION RULES:
1. STANCE NEUTRALITY: Evaluate *behavior and language*, not beliefs. Both sides of an argument must be held to the exact same behavioral standard.
2. CONTEXT MATTERS: Sarcasm, idioms, and hyperbolic jokes may trigger the classifier but are not violations when the target is a general claim, a public group, or the argument itself.
3. SUBTLE HOSTILITY COUNTS: In a debate setting, condescension, mocking sarcasm, bad-faith accusations, strawmanning, and dismissive framing are toxic even when the surface language is polite. Tone and framing carry the signal.

RUBRIC & CATEGORIES:
Assign one of the following categories and a point value.

- "fp" (False Positive) | 0 points
  The flagged words are benign, hyperbolic, joking, or academic. The comment engages the substance of the argument without targeting the opponent personally and without using hostile framing.
  issue_type: "benign_context"

- "flare" (Heated but Substantive Critique) | 1 to 2 points
  The comment is harsh, blunt, or directly negative about the *argument itself* — saying it is wrong, unsupported, outdated, or mistaken. It engages with the substance of the claim. It does NOT include condescension about the opponent, mocking sarcasm, or accusations of bad faith.
  issue_type: "heated_debate"

- "toxic" (Ad Hominem, Abuse, or Subtle Hostility in Debate) | 3 to 5 points
  The comment is destructive to healthy debate through any of the following — including cases where the surface language is polite:
    - personal attacks or insults directed at the opponent
    - condescension: patronizing the opponent, questioning their capacity to understand ("I see you still can't quite grasp this")
    - mocking sarcasm used to belittle the opponent rather than illustrate a point ("I love how eager you are to...")
    - bad-faith accusations without specific evidence ("you're dodging", "you're moving the goalposts", "you're arguing in bad faith")
    - strawmanning: visibly misrepresenting the opponent's stated position
    - dismissive framing: signaling the opponent's argument is not worth engaging with ("this is not a serious argument")
    - deadlock behavior: restating the same point without engaging counter-points
  issue_type must be one of:
    - "personal_attack"
    - "dismissive_condescending"
    - "deadlock_loop"
    - "weak_engagement"
    - "bad_faith_accusation"
    - "mocking_sarcasm"
    - "strawman"

- "zero-tolerance" (Severe Violation) | 100 points
  Hate speech, extreme slurs, threats of real-world violence, doxing, encouragement of self-harm.
  issue_type: "severe_explicit_hate"

FEW-SHOT EXAMPLES:

Example 1 (fp):
Context: Post about dietary preferences. Previous: "I love pineapple on pizza."
User Message: "You are literally killing me with that opinion lol, I'm dying."
Output:
{
  "explanation": "Hyperbolic joke about pizza preference. No hostility toward the opponent, no dismissive framing.",
  "category": "fp",
  "issue_type": "benign_context",
  "points": 0
}

Example 2 (flare):
Context: Economic policy. Previous: "We need to print more money to solve this."
User Message: "That is completely absurd. Your data is 10 years out of date and the argument is financially illiterate."
Output:
{
  "explanation": "Harsh but directed at the data and argument ('absurd', 'out of date', 'financially illiterate' describe the argument, not the person). Engages substance.",
  "category": "flare",
  "issue_type": "heated_debate",
  "points": 1
}

Example 3 (toxic - personal attack, explicit):
Context: Local elections. Previous: "I think candidate A has better infrastructure plans."
User Message: "You are a braindead moron. He will bankrupt us."
Output:
{
  "explanation": "Direct ad hominem ('braindead moron'). Crosses from critique of position to personal insult.",
  "category": "toxic",
  "issue_type": "personal_attack",
  "points": 5
}

Example 4 (toxic - subtle, bad-faith accusation with mocking sarcasm):
Context: Abortion debate. Previous: "Abortion bans disproportionately harm low-income women."
User Message: "I love how eager you are to 'refocus' this. You're still dodging the real question though."
Output:
{
  "explanation": "Opens with mocking sarcasm ('I love how eager you are to') to belittle the opponent's framing, then accuses them of 'dodging' without substantive engagement. Subtle toxic behavior despite polite surface language.",
  "category": "toxic",
  "issue_type": "mocking_sarcasm",
  "points": 3
}

Example 5 (toxic - condescension):
Context: Abortion debate. Previous: "Viability is not the only relevant criterion."
User Message: "Okay? You still can't quite get your head around it, I see. Let's try this: if you cannot..."
Output:
{
  "explanation": "'You still can't quite get your head around it' questions the opponent's capacity to understand — textbook condescension. 'Let's try this' is patronizing framing. Targets the person's intellect, not the argument.",
  "category": "toxic",
  "issue_type": "dismissive_condescending",
  "points": 3
}

Example 6 (flare, not toxic - tight call):
Context: Climate policy. Previous: "Nuclear should be part of the transition."
User Message: "No, that's wrong. Nuclear has a waste problem your comment ignores, and the cost numbers you're implying don't match recent data."
Output:
{
  "explanation": "Direct disagreement engaging the two specific points (waste, cost numbers). No condescension, no bad-faith accusation, no mockery. Stays on the argument.",
  "category": "flare",
  "issue_type": "heated_debate",
  "points": 1
}

TASK:
Analyze the following inputs and output a valid JSON object matching the requested schema. Provide the explanation first, then category, then issue_type, then points.
"""


class DebateAwareReasoner:
    """LLM-as-judge variant tuned for debate-setting subtle hostility.

    Drop-in substitute for models.LLMReasoner in evaluation pipelines.
    Not used by the simulation orchestrator.
    """

    def __init__(self, model: str = "llama3.2:3b") -> None:
        self.model: str = model
        _ = ollama.pull(self.model)

    def analyze_intent(
        self, comment_body: str, parent_body: str, thread_context: str
    ) -> ReasoningResult:
        user_message = (
            "TASK:\n"
            "Analyze the following inputs and output a valid JSON object matching the requested schema.\n\n"
            f"Original Post (topic):\n{thread_context}\n\n"
            f"Previous Comment (immediate context):\n{parent_body}\n\n"
            f"User Message:\n{comment_body}\n"
        )
        messages = [
            {"role": "system", "content": DEBATE_SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
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
            explanation="Debate judge output was malformed; applied conservative fallback.",
            category="toxic",
            issue_type="unclassified",
            points=3,
        )
