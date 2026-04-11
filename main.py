import random

from filtering import ThreadFilter
from interfaces import InterventionResult, ReasoningResult, RedditThread
from models import ToxicityClassifier
from orchestrator import ModerationOrchestrator
from visualization import visualize_graph


class DummyReasoner:
    def analyze_intent(
        self, text: str, parent_text: str, root_context: str
    ) -> ReasoningResult:
        roll = random.random()
        if roll > 0.9:
            return {
                "category": "zero-tolerance",
                "points": 10,
                "explanation": "Severe attack.",
            }
        elif roll > 0.4:
            return {
                "category": "toxic",
                "points": random.randint(2, 5),
                "explanation": "Hostile tone.",
            }
        else:
            return {
                "category": "flare",
                "points": 0,
                "explanation": "Just heated debate.",
            }


class DummyIntervener:
    def generate_intervention(
        self, text: str, author: str, infractions: int
    ) -> InterventionResult:
        return {
            "intervention_text": "Please keep the discussion civil.",
            "tone_used": "neutral",
        }


if __name__ == "__main__":
    classifier = ToxicityClassifier()
    filter_engine = ThreadFilter(classifier, max_threads=2, chain_length=4)
    target_threads: list[RedditThread] = filter_engine.filter_file(
        "data.jsonl", "out.jsonl"
    )

    orchestrator = ModerationOrchestrator(
        classifier, reasoner=DummyReasoner(), intervener=DummyIntervener()
    )

    for thread in target_threads:
        processed_graph = orchestrator.process_thread(thread)
        visualize_graph(processed_graph)
