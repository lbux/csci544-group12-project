"""Re-run the seed-vs-generated analysis on the three pilot transcripts
using the debate-aware judge (subtle hostility sensitive).

Outputs are written alongside the original analyze_pilots.py results,
with the `_debate_eval` suffix so both versions can be compared.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_pilots import (
    PILOT_FILES,
    detect_format,
    row_author,
    row_kind,
    row_text,
    threshold_sweep,
)
from debate_judge import DebateAwareReasoner
from evaluate import LocalClassifier, load_jsonl, root_context_for


CLASSIFIER_PATH = "cga_deberta_final"
JUDGE_MODEL = "llama3.2:3b"
OUT_DIR = Path("eval_records")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    classifier = LocalClassifier(CLASSIFIER_PATH)
    judge = DebateAwareReasoner(model=JUDGE_MODEL)

    all_summaries: dict[str, Any] = {}

    for path_str in PILOT_FILES:
        path = Path(path_str)
        if not path.exists():
            print(f"[skip] {path}")
            continue
        rows = load_jsonl(path)
        if not rows:
            continue
        fmt = detect_format(rows[0])
        root_ctx = root_context_for(rows, fmt)

        out_path = OUT_DIR / f"{path.stem}_debate_eval.jsonl"
        entries: list[dict[str, Any]] = []
        prev = ""
        print(f"\n=== {path.name} ({fmt}) [DEBATE JUDGE] ===")
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
                    "source_file": path.name,
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

        seed_entries = [e for e in entries if e["kind"] == "seed"]
        gen_entries = [e for e in entries if e["kind"] == "generated"]
        med_entries = [e for e in entries if e["kind"] == "mediator"]

        def cat_counts(es: list[dict[str, Any]]) -> dict[str, int]:
            return dict(Counter(e["judge"]["category"] for e in es))

        def max_rob(es: list[dict[str, Any]]) -> float:
            return max((e["roberta_toxicity"] for e in es), default=0.0)

        summary: dict[str, Any] = {
            "seed": {
                "n": len(seed_entries),
                "categories": cat_counts(seed_entries),
                "points_total": sum(e["judge"]["points"] for e in seed_entries),
                "roberta_max": max_rob(seed_entries),
            },
            "generated": {
                "n": len(gen_entries),
                "categories": cat_counts(gen_entries),
                "points_total": sum(e["judge"]["points"] for e in gen_entries),
                "roberta_max": max_rob(gen_entries),
            },
            "mediator_turns_in_transcript": len(med_entries),
            "threshold_sweep_generated_only": threshold_sweep(entries),
        }
        all_summaries[path.name] = summary
        print(f"\nsummary: {json.dumps(summary, indent=2)}")

    out_summary = OUT_DIR / "pilots_debate_eval_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n\nCross-transcript summary -> {out_summary}")


if __name__ == "__main__":
    main()
