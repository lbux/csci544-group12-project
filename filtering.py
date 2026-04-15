import json
from typing import cast

from interfaces import Comment, RedditThread, ToxicityScorer


class ThreadFilter:
    def __init__(
        self,
        classifier: ToxicityScorer,
        max_threads: int = 5,
        chain_threshold: float = 0.75,
        chain_length: int = 2,
    ) -> None:
        self.classifier: ToxicityScorer = classifier
        self.max_threads: int = max_threads
        self.chain_threshold: float = chain_threshold
        self.chain_length: int = chain_length

    def _enrich_and_score_tree(self, comments: list[Comment]) -> None:
        """Recursively scores and adds the 'toxicity' key to every comment in place."""
        for c in comments:
            if "toxicity" not in c:
                c["toxicity"] = self.classifier.predict(text=c["body"])
            if c.get("replies"):
                self._enrich_and_score_tree(comments=c["replies"])

    def has_toxic_chain(self, comments: list[Comment], current_streak: int = 0) -> bool:
        """Checks if there is a continuous chain of replies exceeding the threshold."""
        for c in comments:
            score: float = c.get("toxicity", 0.0)
            streak: int = current_streak + 1 if score >= self.chain_threshold else 0
            if streak >= self.chain_length:
                return True
            if self.has_toxic_chain(c.get("replies", []), streak):
                return True
        return False

    # While this would have been cool, the model either scores posts at 0.0 or 0.95-99 most of the time
    # def has_high_escalation(self, comments: list[Comment], parent_score: float = 0.0) -> bool:
    #     """Finds if any reply spikes drastically in toxicity compared to its parent."""
    #     for c in comments:
    #         score: float = c.get("toxicity", 0.0)

    #         if (score - parent_score) >= self.escalation_delta:
    #             return True
    #         if self.has_high_escalation(c.get("replies", []), score):
    #             return True
    #     return False

    def filter_file(self, input_jsonl: str, output_jsonl: str) -> list[RedditThread]:
        """Filters the JSONL dump into an output file that meet the selectrin critera as defined by the class initialization"""
        selected_threads: list[RedditThread] = []

        with (
            open(input_jsonl, "r", encoding="utf-8") as infile,
            open(output_jsonl, "w", encoding="utf-8") as outfile,
        ):
            for line in infile:
                if len(selected_threads) >= self.max_threads:
                    break

                thread_data: RedditThread = cast(RedditThread, json.loads(line))

                post_text: str = thread_data.get("body", "") or thread_data.get(
                    "title", ""
                )
                if "body_toxicity" not in thread_data:
                    thread_data["body_toxicity"] = self.classifier.predict(post_text)

                self._enrich_and_score_tree(thread_data["comments"])

                is_chain: bool = self.has_toxic_chain(thread_data["comments"])

                if is_chain:
                    print(f"Selected Thread: {thread_data['submission_id']}")
                    selected_threads.append(thread_data)
                    _ = outfile.write(json.dumps(thread_data) + "\n")
            return selected_threads
