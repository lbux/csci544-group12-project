from __future__ import annotations

import json

import networkx as nx
from networkx.classes.digraph import DiGraph

from interfaces import (
    InterventionAgent,
    InterventionResult,
    ReasoningAgent,
    ReasoningResult,
    RedditThread,
)
from utils import flatten_comments


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


class ModerationOrchestrator:
    def __init__(
        self,
        reasoner: ReasoningAgent,
        intervener: InterventionAgent,
        intervention_threshold: int = 10,
        toxicity_threshold: float = 0.6,
    ) -> None:
        self.reasoner: ReasoningAgent = reasoner
        self.intervener: InterventionAgent = intervener
        self.intervention_threshold: int = intervention_threshold
        self.toxicity_threshold: float = toxicity_threshold
        self.tracker: ThreadStateTracker = ThreadStateTracker()
        self.graph: DiGraph[str] = nx.DiGraph()
        self.current_thread_id: str = ""

    def process_thread(self, thread: RedditThread) -> nx.DiGraph[str]:
        """Converts a thread to a graph and processes it chronologically to allow for scoring comments as they \"arrive\""""
        self.graph.clear()
        self.tracker = ThreadStateTracker()
        self.current_thread_id = thread["submission_id"]

        submission_id = thread["submission_id"]
        post_body = thread.get("selftext", "").strip()
        if post_body:
            thread_context = f"Title: {thread['title']}\nPost: {post_body}"
        else:
            thread_context = f"Title: {thread['title']}"

        self.graph.add_node(
            submission_id, body=thread_context, author=thread["author"], type="post"
        )

        flat_comments = list(flatten_comments(thread["comments"], submission_id))
        flat_comments.sort(key=lambda x: x[0].get("created_utc", 0.0))

        for comment, parent_id in flat_comments:
            self._ingest_comment(
                comment_id=comment["id"],
                author=comment["author"],
                comment_body=comment["body"],
                toxicity=comment.get("toxicity", 0.0),
                parent_id=parent_id,
                thread_context=thread_context,
            )

        return self.graph

    def _ingest_comment(
        self,
        comment_id: str,
        author: str,
        comment_body: str,
        toxicity: float,
        parent_id: str,
        thread_context: str,
    ) -> None:
        """Internal logic that handles adding nodes to graph, scoring, and intervening"""
        if not author or author in ["[deleted]", "[removed]"]:
            author = "Deleted"

        if parent_id not in self.graph:
            self.graph.add_node(
                parent_id,
                body="[Missing]",
                author="Unavailable",
                type="comment",
                toxicity_score=0.0,
            )

        self.graph.add_node(
            comment_id, body=comment_body, author=author, type="comment"
        )
        _ = self.graph.add_edge(parent_id, comment_id)
        parent_body: str = self.graph.nodes[parent_id].get("body", "")  # pyright: ignore[reportAny]

        self.graph.nodes[comment_id]["toxicity_score"] = toxicity

        if toxicity < self.toxicity_threshold:
            return

        reasoning: ReasoningResult = self.reasoner.analyze_intent(
            comment_body, parent_body, thread_context
        )
        issue_type = (
            "severe_explicit_hate"
            if reasoning.category == "zero-tolerance"
            else reasoning.issue_type
        )

        is_judged_toxic = reasoning.category in ["toxic", "zero-tolerance"]
        cooldown_was_active = self.tracker.in_cooldown()

        thread_penalty = self.tracker.thread_penalty
        if is_judged_toxic:
            thread_penalty = self.tracker.add_penalty(reasoning.points)
            print(
                f"[WARN] Thread {self.current_thread_id} gained {reasoning.points} pts (Total: {thread_penalty})"
            )

        self.graph.nodes[comment_id]["issue_type"] = issue_type
        self.graph.nodes[comment_id]["reasoning_category"] = reasoning.category
        self.graph.nodes[comment_id]["reasoning_explanation"] = reasoning.explanation
        self.graph.nodes[comment_id]["points_assigned"] = reasoning.points
        self.graph.nodes[comment_id]["thread_cumulative_penalty"] = thread_penalty

        log_entry = {
            "thread_id": self.current_thread_id,
            "comment_id": comment_id,
            "author": author,
            "toxicity_score": toxicity,
            "category": reasoning.category,
            "issue_type": issue_type,
            "points_assigned": reasoning.points,
            "thread_cumulative_penalty": thread_penalty,
        }
        with open("reasoning_logs.jsonl", "a", encoding="utf-8") as f:
            _ = f.write(json.dumps(log_entry) + "\n")

        should_intervene = False
        if issue_type == "severe_explicit_hate" and is_judged_toxic:
            should_intervene = True
        elif is_judged_toxic and thread_penalty >= self.intervention_threshold:
            if not cooldown_was_active:
                should_intervene = True

        if should_intervene:
            action: InterventionResult = self.intervener.generate_intervention(
                text=comment_body,
                author=author,
                cumulative_penalty=thread_penalty,
                parent_text=parent_body,
                root_context=thread_context,
                issue_type=issue_type,
                reasoning_explanation=reasoning.explanation,
            )
            intervention_id = f"intervention:{comment_id}:{thread_penalty}"
            self.graph.add_node(
                intervention_id,
                author="Mediator",
                type="intervention",
                strategy=action["strategy"],
                target=action["target"],
                tone_used=action["tone_used"],
                rationale=action["rationale"],
                body=action["intervention_text"],
                text=action["intervention_text"],
                issue_type=issue_type,
                points_assigned=reasoning.points,
                thread_cumulative_penalty=thread_penalty,
            )
            _ = self.graph.add_edge(comment_id, intervention_id)

            intervention_log = {
                "thread_id": self.current_thread_id,
                "comment_id": comment_id,
                "author": author,
                "issue_type": issue_type,
                "strategy": action["strategy"],
                "target": action["target"],
                "rationale": action["rationale"],
                "intervention_text": action["intervention_text"],
                "thread_cumulative_penalty": thread_penalty,
            }
            with open("intervention_logs.jsonl", "a", encoding="utf-8") as f:
                _ = f.write(json.dumps(intervention_log) + "\n")

            print(
                f"[INTERVENTION - {action['tone_used'].upper()}] Thread {self.current_thread_id}: {action['intervention_text']}"
            )

            self.tracker.reset_penalty()
            self.tracker.start_cooldown(2)

        if cooldown_was_active and is_judged_toxic and not should_intervene:
            self.tracker.tick_cooldown()
