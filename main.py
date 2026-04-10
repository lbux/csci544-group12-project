import random

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx

from filtering import ThreadFilter
from interfaces import InterventionResult, ReasoningResult, RedditThread
from models import ToxicityClassifier
from orchestrator import ModerationOrchestrator


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


def visualize_graph(G: nx.DiGraph) -> None:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    _ = plt.figure(figsize=(18, 12))  # pyright: ignore[reportUnknownMemberType]

    pos = nx.spring_layout(  # pyright: ignore[reportUnknownVariableType]
        G,  # pyright: ignore[reportUnknownArgumentType]
        k=1.2,
        iterations=100,
        seed=42,
    )

    cmap = plt.get_cmap("Reds")

    node_colors: list[str] = []
    node_sizes: list[int] = []
    # double check this data type
    labels: dict[str, str] = {}

    for node, data in G.nodes(data=True):  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAny]
        score: float = data.get("toxicity_score", 0.0)  # pyright: ignore[reportAny]

        # Original post
        if data.get("type") == "post":  # pyright: ignore[reportAny]
            node_colors.append("gold")
            node_sizes.append(3500)
            labels[node] = f"Original Post\n{data.get('author')}"  # pyright: ignore[reportAny]

        # Standardize Comments
        else:
            node_colors.append(mcolors.to_hex(cmap(score)))
            node_sizes.append(2500)

            author: str = data.get("author", "Unknown")  # pyright: ignore[reportAny]
            if len(author) > 8:
                author = author[:5] + "..."

            labels[node] = f"{author}\nTox: {score:.2f}"

    nx.draw_networkx(
        G,  # pyright: ignore[reportUnknownArgumentType]
        pos,  # pyright: ignore[reportUnknownArgumentType]
        labels=labels,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=8,
        font_weight="bold",
        edgecolors="black",
        linewidths=1.5,
        arrowsize=20,
        edge_color="gray",
    )

    _ = plt.title("Thread Result", fontsize=16, fontweight="bold")  # pyright: ignore[reportUnknownMemberType]
    _ = plt.axis("off")  # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    classifier = ToxicityClassifier()

    # 1. Filter JSONL files
    filter_engine = ThreadFilter(classifier, threshold=0.1, max_threads=2)
    target_threads: list[RedditThread] = filter_engine.filter_file("data.jsonl")

    # 2. Run Orchestrator
    orchestrator = ModerationOrchestrator(
        classifier, reasoner=DummyReasoner(), intervener=DummyIntervener()
    )

    for thread in target_threads:
        print(f"\n--- Processing Thread: {thread['title']} ---")
        processed_graph = orchestrator.process_thread(thread)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        visualize_graph(processed_graph)
