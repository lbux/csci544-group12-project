from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
from networkx.classes.digraph import DiGraph

from interfaces import (
    InterventionAgent,
    InterventionResult,
    ReasoningAgent,
    ReasoningResult,
    RedditThread,
    ToxicityClassifier,
)
from utils import flatten_comments


class UserStateTracker:
    def __init__(self) -> None:
        # Maps username to accumulated penalty points
        self.user_penalties: dict[str, int] = defaultdict(int)

    def add_penalty(self, author: str, points: int) -> int:
        """Adds points and returns the user's new total"""
        self.user_penalties[author] += points
        return self.user_penalties[author]

    def reset_penalty(self, author: str) -> None:
        """Optionally clear the user's penalty score after an intervention"""
        self.user_penalties[author] = 0


class ModerationOrchestrator:
    def __init__(
        self,
        classifier: ToxicityClassifier,
        reasoner: ReasoningAgent,
        intervener: InterventionAgent,
        intervention_threshold: int = 10,
    ) -> None:
        self.classifier: ToxicityClassifier = classifier
        self.reasoner: ReasoningAgent = reasoner
        self.intervener: InterventionAgent = intervener
        self.intervention_threshold: int = intervention_threshold
        self.tracker: UserStateTracker = UserStateTracker()
        self.graph: DiGraph[Any] = nx.DiGraph()  # pyright: ignore[reportExplicitAny]

    def process_thread(self, thread: RedditThread) -> nx.DiGraph[str]:
        """Converts a thread to a graph and processes it chronologically to allow for scoring comments as they \"arrive\""""
        self.graph.clear()

        root_id = thread["submission_id"]
        post_body = thread.get("body", "")
        root_context = f"Title: {thread['title']}\nPost: {post_body}"

        self.graph.add_node(
            root_id, text=root_context, author=thread["author"], type="post"
        )

        flat_comments = list(flatten_comments(thread["comments"], root_id))
        flat_comments.sort(key=lambda x: x[0].get("created_utc", 0.0))

        for comment, parent_id in flat_comments:
            self._ingest_comment(
                comment["id"],
                comment["author"],
                comment["body"],
                parent_id,
                root_context,
            )

        return self.graph

    def _ingest_comment(
        self, node_id: str, author: str, text: str, parent_id: str, root_context: str
    ) -> None:
        """Internal logic that handles adding nodes to graph, scoring, and intervening"""
        if not author or author == "[deleted]":
            author = "Deleted"

        # To not break NetworkX, we add generic nodes for when a comment relies on a parent
        # that does not exist (reddit api didn't provide it, network issues, banned account, etc)
        if parent_id not in self.graph:
            self.graph.add_node(
                parent_id,
                text="[Missing]",
                author="Unavailable",
                type="comment",
                toxicity_score=0.0,
            )

        self.graph.add_node(node_id, text=text, author=author, type="comment")
        _ = self.graph.add_edge(parent_id, node_id)
        parent_text: str = self.graph.nodes[parent_id].get("text", "")  # pyright: ignore[reportAny]

        score: float = self.classifier.predict(text)
        self.graph.nodes[node_id]["toxicity_score"] = score

        # TODO: Add an actual threshold that works for our data. This is a generic placeholder.
        if score >= 0.6:
            # This would be the result of the agent. We have a category for type of toxicity, how many points to
            # penalize, and the reasoning from the agent.
            reasoning: ReasoningResult = self.reasoner.analyze_intent(
                text, parent_text, root_context
            )

            if reasoning["category"] in ["toxic", "zero-tolerance"]:
                # If the model categorizes as the comment as anything other than a heated flare, penalize
                # The categories would be something we define. We can assign a rubric to the model for how to
                # assign points
                current_penalty = self.tracker.add_penalty(author, reasoning["points"])
                print(
                    f"[WARN] {author} gained {reasoning['points']} pts (Total: {current_penalty})"
                )

                # If the user's points reach the threshold, intervene.
                if current_penalty >= self.intervention_threshold:
                    action: InterventionResult = self.intervener.generate_intervention(
                        text, author, current_penalty
                    )
                    print(
                        f"[INTERVENTION - {action['tone_used'].upper()}] To {author}: {action['intervention_text']}"
                    )

                    # Potentially reset the points after the intervention?
                    self.tracker.reset_penalty(author)
