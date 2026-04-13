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
    body: NotRequired[str]
    created_utc: float
    comments: list[Comment]
    body_toxicity: NotRequired[float]


class ReasoningResult(BaseModel):
    explanation: str
    category: str  # e.g., "flare", "toxic", "zero-tolerance"
    points: int


class InterventionResult(TypedDict):
    intervention_text: str
    tone_used: str


class ToxicityScorer(Protocol):
    def predict(self, text: str) -> float: ...


class ReasoningAgent(Protocol):
    def analyze_intent(
        self, text: str, parent_text: str, root_context: str
    ) -> ReasoningResult: ...


class InterventionAgent(Protocol):
    def generate_intervention(
        self, text: str, author: str, infractions: int
    ) -> InterventionResult: ...


class CounterfactualSimulator(Protocol):
    def run_simulation(
        self, current_graph: object, proposed_intervention: str
    ) -> float: ...
