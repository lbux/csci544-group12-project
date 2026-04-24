"""Run the debate-aware judge + local CGA classifier over every transcript
in `sim_debate_records/<model>/`, writing one `*_debate_eval.jsonl` per
input file into `eval_records/<model>/`.

Mirrors `analyze_pilots_debate.py` but is generalized to all model
folders. Skips files whose eval output already exists unless --force.

Usage:
    uv run python eval_sim_debates.py
    uv run python eval_sim_debates.py --models gemma4_e4b gemma4_uncensored
    uv run python eval_sim_debates.py --force
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_pilots import detect_format, row_author, row_kind, row_text, threshold_sweep
from debate_judge import DebateAwareReasoner
from evaluate import LocalClassifier, load_jsonl, root_context_for


CLASSIFIER_PATH = "cga_deberta_final"
JUDGE_MODEL = "llama3.2:3b"
SIM_ROOT = Path("sim_debate_records")
OUT_ROOT = Path("eval_records")


def discover_transcripts(models: list[str] | None) -> list[Path]:
    if not SIM_ROOT.exists():
        return []
    paths: list[Path] = []
    for sub in sorted(SIM_ROOT.iterdir()):
        if not sub.is_dir() or sub.name == "logs":
            continue
        if models and sub.name not in models:
            continue
        paths.extend(sorted(sub.glob("*.jsonl")))
    return paths


def eval_one(
    input_path: Path,
    out_path: Path,
    classifier: LocalClassifier,
    judge: DebateAwareReasoner,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    if not rows:
        return {}
    fmt = detect_format(rows[0])
    root_ctx = root_context_for(rows, fmt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    prev = ""
    print(f"\n=== {input_path.parent.name}/{input_path.name} ({fmt}) ===")
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            kind = row_kind(row, fmt)
            text = row_text(row, fmt)
            author = row_author(row, fmt)
            if not text:
                prev = text
                continue
            rob = float(classifier.predict(text))
            result = judge.analyze_intent(
                comment_body=text,
                parent_body=prev,
                thread_context=root_ctx,
            )
            entry = {
                "source_file": input_path.name,
                "model_dir": input_path.parent.name,
                "row_idx": i,
                "kind": kind,
                "author": author,
                "text": text,
                "roberta_toxicity": rob,
                "judge": {
                    "category": result.category,
                    "issue_type": result.issue_type,
                    "points": result.points,
                    "explanation": result.explanation,
                },
            }
            entries.append(entry)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                f"  {i:>2d} {kind:<10} rob={rob:.4f} "
                f"-> {result.category}/{result.issue_type} ({result.points}p)",
                flush=True,
            )
            prev = text

    seed_es = [e for e in entries if e["kind"] == "seed"]
    gen_es = [e for e in entries if e["kind"] == "generated"]
    med_es = [e for e in entries if e["kind"] == "mediator"]

    def cats(es: list[dict[str, Any]]) -> dict[str, int]:
        return dict(Counter(e["judge"]["category"] for e in es))

    def max_rob(es: list[dict[str, Any]]) -> float:
        return max((e["roberta_toxicity"] for e in es), default=0.0)

    summary: dict[str, Any] = {
        "seed": {
            "n": len(seed_es),
            "categories": cats(seed_es),
            "points_total": sum(e["judge"]["points"] for e in seed_es),
            "roberta_max": max_rob(seed_es),
        },
        "generated": {
            "n": len(gen_es),
            "categories": cats(gen_es),
            "points_total": sum(e["judge"]["points"] for e in gen_es),
            "roberta_max": max_rob(gen_es),
        },
        "mediator_turns_in_transcript": len(med_es),
        "threshold_sweep_generated_only": threshold_sweep(entries),
    }
    print(f"summary: {json.dumps(summary)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model subfolder names under sim_debate_records/. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if the *_debate_eval.jsonl already exists.",
    )
    args = parser.parse_args()

    transcripts = discover_transcripts(args.models)
    if not transcripts:
        print("No transcripts found.")
        return

    classifier = LocalClassifier(CLASSIFIER_PATH)
    judge = DebateAwareReasoner(model=JUDGE_MODEL)

    all_summaries: dict[str, dict[str, Any]] = {}
    for path in transcripts:
        out_dir = OUT_ROOT / path.parent.name
        out_path = out_dir / f"{path.stem}_debate_eval.jsonl"
        key = f"{path.parent.name}/{path.name}"
        if out_path.exists() and not args.force:
            print(f"[skip exists] {key} -> {out_path}")
            continue
        summary = eval_one(path, out_path, classifier, judge)
        if summary:
            all_summaries[key] = summary

    if all_summaries:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        out_summary = OUT_ROOT / "sim_debates_eval_summary.json"
        existing: dict[str, Any] = {}
        if out_summary.exists():
            try:
                existing = json.loads(out_summary.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        existing.update(all_summaries)
        out_summary.write_text(
            json.dumps(existing, indent=2), encoding="utf-8"
        )
        print(f"\nCross-transcript summary -> {out_summary}")


if __name__ == "__main__":
    main()
