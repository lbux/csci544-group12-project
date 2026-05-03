# pyright: reportAny=false
import json
from collections import defaultdict
from pathlib import Path


def summarize_evaluations(eval_dir: str | Path = "eval_records") -> None:
    """Aggregates scores from all evaluations to compare simulation types."""
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        print(f"No evaluation records found in {eval_path}.")
        return

    metrics: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    win_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for file in eval_path.glob("eval_*.jsonl"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        sim_type = "unknown"
        if "naive_" in file.name:
            sim_type = "Naive"
        elif "moderated_reddit_" in file.name:
            sim_type = "Moderated"
        elif "reddit_" in file.name:
            sim_type = "Reddit Aligned"

        metrics[sim_type]["alignment"].append(data.get("alignment_score", 0))
        metrics[sim_type]["quality"].append(data.get("argument_quality", 0))
        metrics[sim_type]["toxicity"].append(data.get("toxicity_level", 0))

        winner = data.get("winner") or "Tie"
        win_counts[sim_type][winner] += 1

    print("\n" + "=" * 50)
    print(" 📊 FINAL DEBATE SIMULATION EXPERIMENT SUMMARY")
    print("=" * 50)

    if not metrics:
        print("No valid evaluation data found.")
        return

    for sim_type, scores in metrics.items():
        avg_align = sum(scores["alignment"]) / len(scores["alignment"])
        avg_qual = sum(scores["quality"]) / len(scores["quality"])
        avg_tox = sum(scores["toxicity"]) / len(scores["toxicity"])

        print(f"\n--- {sim_type} Debates (n={len(scores['alignment'])}) ---")
        print(f"Avg Alignment Score:  {avg_align:.2f} / 10")
        print(f"Avg Argument Quality: {avg_qual:.2f} / 10")
        print(f"Avg Toxicity Level:   {avg_tox:.2f} / 10")

        print("Win Distribution:")
        for winner, count in win_counts[sim_type].items():
            print(f"  - {winner}: {count}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    summarize_evaluations()
