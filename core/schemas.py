from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field


class Comment(TypedDict):
    id: str
    author: str
    body: str
    created_utc: float
    replies: list["Comment"]
    round_idx: NotRequired[int]
    toxicity: NotRequired[float]
    toxicity_classifier: NotRequired[str]
    generated: NotRequired[bool]
    type: NotRequired[str]
    parent_id: NotRequired[str]
    # Moderation specific fields
    judge: NotRequired[Any]  # pyright: ignore[reportExplicitAny]
    issue_type: NotRequired[str]
    thread_cumulative_penalty: NotRequired[int]
    strategy: NotRequired[str]
    target: NotRequired[str]
    tone_used: NotRequired[str]
    rationale: NotRequired[str]
    points_assigned: NotRequired[int]


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


class ReasoningResult(BaseModel):
    explanation: str = Field(description="Explanation for the classification.")
    category: str = Field(
        description="Must be one of: 'fp', 'flare', 'toxic', or 'zero-tolerance'."
    )
    issue_type: str = Field(
        description="Specific issue type, e.g., 'personal_attack', 'benign_context', etc."
    )
    points: int = Field(description="Penalty points assigned based on the category.")


class InterventionResult(BaseModel):
    strategy: str = Field(
        description="Chosen intervention strategy (e.g., 'tone_reset', 'claim_refocusing')."
    )
    target: str = Field(
        description="Must be one of 'author', 'both', 'thread', or 'moderation_review'."
    )
    tone_used: str = Field(description="Always set to 'neutral'.")
    rationale: str = Field(description="Why this strategy was chosen.")
    intervention_text: str = Field(description="The 1-2 sentence intervention message.")


class DebateEvaluation(BaseModel):
    alignment_score: int = Field(
        description="Score 1-10 on how well agents stayed in persona/stance."
    )
    argument_quality: int = Field(
        description="Score 1-10 on logical soundness and engagement."
    )
    toxicity_level: int = Field(
        description="Score 1-10 on overall thread toxicity (1=civil, 10=abusive)."
    )
    winner: Literal["Agent 1", "Agent 2", "Tie"] = Field(
        description="You must strictly choose 'Agent 1', 'Agent 2', or 'Tie'."
    )
    rationale: str = Field(description="A 2-3 sentence explanation for these scores.")
