"""Score every post body and comment in a scraped Reddit jsonl with the
fine-tuned CGA classifier (Tier 1). Writes a flat jsonl of scored entries
plus a top-K file ready to feed into the Tier 2 judge.

Usage:
    uv run python score_reddit.py \\
        --input data/reddit_abortiondebate.jsonl \\
        --output eval_records/reddit_abortiondebate_scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from evaluate import LocalClassifier


def iter_comments(comments: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for c in comments:
        yield c
        replies = c.get("replies")
        if replies:
            yield from iter_comments(replies)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Scraped Reddit jsonl.")
    parser.add_argument(
        "--output",
        required=True,
        help="Flat jsonl of scored entries (one per post/comment).",
    )
    parser.add_argument(
        "--classifier",
        default="cga_deberta_final",
        help="Path to local fine-tuned classifier.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Also save top-K highest-scored entries for Tier 2 follow-up.",
    )
    parser.add_argument(
        "--top-k-output",
        default=None,
        help="Path for top-K file (default: <output>.top<K>.jsonl).",
    )
    return parser.parse_args()


def summarize(scores: list[float]) -> None:
    n = len(scores)
    if n == 0:
        print("No scores collected.")
        return

    print(f"\n=== distribution over n={n} entries ===")
    print(f"mean   = {statistics.mean(scores):.4f}")
    print(f"median = {statistics.median(scores):.4f}")
    print(f"stdev  = {statistics.stdev(scores):.4f}" if n > 1 else "stdev  = n/a")
    print(f"min    = {min(scores):.4f}")
    print(f"max    = {max(scores):.4f}")

    print("\n=== threshold counts ===")
    for thresh in (0.3, 0.5, 0.6, 0.8, 0.9, 0.95):
        c = sum(1 for s in scores if s >= thresh)
        print(f">= {thresh:.2f}:  {c:>6d}  ({100 * c / n:5.2f}%)")

    print("\n=== histogram (10 bins over [0, 1]) ===")
    bins = [i / 10 for i in range(11)]
    for lo, hi in zip(bins, bins[1:]):
        c = sum(1 for s in scores if lo <= s < hi)
        bar = "#" * min(60, int(60 * c / n))
        print(f"[{lo:.1f}, {hi:.1f})  {c:>6d}  {bar}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading classifier from {args.classifier} ...")
    clf = LocalClassifier(args.classifier)

    scored: list[dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            thread = json.loads(line)
            submission_id = thread.get("submission_id")

            title = (thread.get("title") or "").strip()
            selftext = (thread.get("selftext") or "").strip()
            post_text = (
                f"{title}\n\n{selftext}".strip() if selftext else title
            )
            if post_text:
                entry = {
                    "submission_id": submission_id,
                    "type": "post",
                    "id": submission_id,
                    "author": thread.get("author"),
                    "body": post_text,
                    "toxicity": clf.predict(post_text),
                }
                scored.append(entry)
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

            for c in iter_comments(thread.get("comments", [])):
                body = (c.get("body") or "").strip()
                if not body:
                    continue
                entry = {
                    "submission_id": submission_id,
                    "type": "comment",
                    "id": c.get("id"),
                    "author": c.get("author"),
                    "body": body,
                    "toxicity": clf.predict(body),
                }
                scored.append(entry)
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if line_no % 10 == 0:
                print(
                    f"  processed {line_no} threads, "
                    f"{len(scored)} entries scored",
                    flush=True,
                )

    scores = [e["toxicity"] for e in scored]
    summarize(scores)

    top_k = args.top_k
    top_output = (
        Path(args.top_k_output)
        if args.top_k_output
        else output_path.with_suffix(f".top{top_k}.jsonl")
    )
    top_output.parent.mkdir(parents=True, exist_ok=True)
    scored_sorted = sorted(scored, key=lambda e: e["toxicity"], reverse=True)
    with open(top_output, "w", encoding="utf-8") as f:
        for e in scored_sorted[:top_k]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nTop-{top_k} highest-scored entries -> {top_output}")
    print(f"Full scored file -> {output_path}")


if __name__ == "__main__":
    main()
