"""Run the Tier 2 LLM judge on a jsonl of already-scored Reddit entries
(produced by score_reddit.py). Writes an augmented jsonl with a ``judge``
field and prints category counts.

Usage:
    uv run python judge_top.py \\
        --input eval_records/reddit_abortiondebate_scored.top50.jsonl \\
        --output eval_records/reddit_abortiondebate_top50_judged.jsonl \\
        --judge-model llama3.2:3b
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from models import LLMReasoner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", default="llama3.2:3b")
    parser.add_argument(
        "--thread-context",
        default="Reddit discussion thread from r/Abortiondebate.",
        help="Topic context passed to the judge for every entry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    judge = LLMReasoner(model=args.judge_model)
    entries = [json.loads(line) for line in open(input_path) if line.strip()]
    print(f"Judging {len(entries)} entries with {args.judge_model} ...")

    cats: Counter[str] = Counter()
    total_points = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, e in enumerate(entries, start=1):
            result = judge.analyze_intent(
                comment_body=e["body"],
                parent_body="",
                thread_context=args.thread_context,
            )
            e["judge"] = {
                "category": result.category,
                "issue_type": result.issue_type,
                "points": result.points,
                "explanation": result.explanation,
            }
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")
            fout.flush()
            cats[result.category] += 1
            total_points += result.points
            print(
                f"[{i:>2d}/{len(entries)}] rob={e['toxicity']:.3f} "
                f"-> {result.category}/{result.issue_type} ({result.points}p)",
                flush=True,
            )

    print("\n=== category counts ===")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:<18}  {n}")
    print(f"total points: {total_points}")
    print(f"mean points/entry: {total_points / len(entries):.2f}")


if __name__ == "__main__":
    main()
