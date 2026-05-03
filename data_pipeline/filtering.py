import json
from pathlib import Path

from core.schemas import Comment, RedditThread


class ThreadFilter:
    def __init__(
        self, max_threads: int = 5, chain_threshold: float = 0.75, chain_length: int = 2
    ) -> None:
        self.max_threads: int = max_threads
        self.chain_threshold: float = chain_threshold
        self.chain_length: int = chain_length

    def has_toxic_chain(self, comments: list[Comment], current_streak: int = 0) -> bool:
        for c in comments:
            score: float = float(c.get("toxicity", 0.0))
            streak: int = current_streak + 1 if score >= self.chain_threshold else 0

            if streak >= self.chain_length:
                return True
            if self.has_toxic_chain(c.get("replies", []), streak):
                return True

        return False

    def run_filtering(
        self, input_path: str | Path, output_path: str | Path
    ) -> list[RedditThread]:
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_path} not found.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected_threads: list[RedditThread] = []

        with (
            open(input_path, "r", encoding="utf-8") as infile,
            open(output_path, "w", encoding="utf-8") as outfile,
        ):
            for line in infile:
                if len(selected_threads) >= self.max_threads:
                    break

                if not line.strip():
                    continue

                thread_data: RedditThread = json.loads(line)  # pyright: ignore[reportAny]

                if self.has_toxic_chain(thread_data["comments"]):
                    print(f"Selected Thread: {thread_data['submission_id']}")
                    selected_threads.append(thread_data)
                    outfile.write(json.dumps(thread_data, ensure_ascii=False) + "\n")  # pyright: ignore[reportUnusedCallResult]

        print(
            f"\nSuccessfully filtered {len(selected_threads)} threads into {output_path}"
        )
        return selected_threads
