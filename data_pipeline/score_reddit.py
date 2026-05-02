import json
from pathlib import Path

from core.moderation import ToxicityClassifier
from core.schemas import Comment, RedditThread


def score_comment_tree(comments: list[Comment], classifier: ToxicityClassifier) -> None:
    """Recursively scores and adds the 'toxicity' key to every comment in place."""
    for c in comments:
        if "toxicity" not in c:
            c["toxicity"] = classifier.predict(c["body"].strip())
        if c.get("replies"):
            score_comment_tree(c["replies"], classifier)


def run_scoring(input_path: str | Path, output_path: str | Path) -> None:
    """Entry point for the Jupyter Notebook to score raw threads."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file {input_path} not found. Did you run the scraper?"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier = ToxicityClassifier()
    processed_count = 0

    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            if not line.strip():
                continue

            thread_data: RedditThread = json.loads(line)  # pyright: ignore[reportAny]

            post_text = thread_data.get("selftext", "") or thread_data["title"]
            if "body_toxicity" not in thread_data:
                thread_data["body_toxicity"] = classifier.predict(post_text.strip())

            score_comment_tree(thread_data["comments"], classifier)

            outfile.write(json.dumps(thread_data, ensure_ascii=False) + "\n")  # pyright: ignore[reportUnusedCallResult]
            processed_count += 1
            print(f"Scored thread {processed_count}: {thread_data['submission_id']}")

    print(f"Finished scoring {processed_count} threads. Saved to {output_path}")
