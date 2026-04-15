from typing import NotRequired, Protocol, TypedDict

from pydantic import BaseModel


class Comment(TypedDict):
    id: str
    author: str
    body: str
    created_utc: float
    replies: list["Comment"]
    toxicity: NotRequired[float]


class RedditThread(TypedDict):
    submission_id: str
    author: str
    title: str
    selftext: NotRequired[str]
    created_utc: float
    comments: list[Comment]
    body_toxicity: NotRequired[float]


class ReasoningResult(BaseModel):
    explanation: str
    category: str  # e.g., "flare", "toxic", "zero-tolerance"
    issue_type: str
    points: int


class InterventionResult(TypedDict):
    strategy: str
    rationale: str
    target: str
    intervention_text: str
    tone_used: str


class ToxicityScorer(Protocol):
    def predict(self, text: str) -> float: ...


class ReasoningAgent(Protocol):
    def analyze_intent(
        self, comment_body: str, parent_body: str, thread_context: str
    ) -> ReasoningResult: ...


class InterventionAgent(Protocol):
    def generate_intervention(
        self,
        text: str,
        author: str,
        cumulative_penalty: int,
        parent_text: str,
        root_context: str,
        issue_type: str,
        reasoning_explanation: str,
    ) -> InterventionResult: ...


class CounterfactualSimulator(Protocol):
    def run_simulation(
        self, current_graph: object, proposed_intervention: str
    ) -> float: ...
