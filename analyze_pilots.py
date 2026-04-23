"""Produce concrete evidence for paper-scope decisions from the three
llama3.1:8b pilot transcripts:

  1. Judge EVERY row (seed + generated), not just generated turns, so we can
     contrast real Reddit turns with LLM turns in the same thread.
  2. Run a post-hoc intervention-threshold sweep over the generated turns of
     each transcript to show how many interventions would fire at
     thresholds {3, 5, 7, 10}.

Writes one rich eval jsonl per transcript and a cross-transcript summary.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate import LocalClassifier, iter_evaluable_turns, load_jsonl, root_context_for
from models import LLMReasoner


CLASSIFIER_PATH = "cga_deberta_final"
JUDGE_MODEL = "llama3.2:3b"

PILOT_FILES = [
    "sim_debate_records/naive_abortion_rights_aece1_llama3.1_8b.jsonl",
    "sim_debate_records/reddit_abortion_rights_1s3uc44_27a61_llama3.1_8b.jsonl",
    "sim_debate_records/moderated_reddit_abortion_rights_1s3uc44_42d1b_llama3.1_8b.jsonl",
]

OUT_DIR = Path("eval_records")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_format(row: dict[str, Any]) -> str:
    if "body" in row and "author" in row and "generated" in row:
        return "reddit"
    if "text" in row and "agent" in row:
        return "naive"
    raise ValueError(f"Unknown row shape: {sorted(row.keys())}")


def row_text(row: dict[str, Any], fmt: str) -> str:
    if fmt == "naive":
        return (row.get("text") or "").strip()
    return (row.get("body") or "").strip()


def row_author(row: dict[str, Any], fmt: str) -> str:
    if fmt == "naive":
        return row.get("agent") or ""
    return row.get("author") or ""


def row_kind(row: dict[str, Any], fmt: str) -> str:
    """seed | generated | mediator"""
    if fmt == "naive":
        return "generated"
    if str(row.get("id") or "").startswith("mediator_"):
        return "mediator"
    if row.get("generated") is True:
        return "generated"
    return "seed"


def threshold_sweep(entries: list[dict[str, Any]], cooldown: int = 2) -> dict[int, int]:
    """For each threshold, walk the generated-turn sequence and count
    how many interventions would have fired. Resets the running penalty
    on each fire and skips `cooldown` subsequent turns from triggering
    again (matching orchestrator semantics)."""
    results: dict[int, int] = {}
    for thresh in (3, 5, 7, 10):
        penalty = 0
        cd = 0
        fires = 0
        for e in entries:
            if e.get("kind") != "generated":
                continue
            pts = e["judge"]["points"]
            judged_toxic = e["judge"]["category"] in ("toxic", "zero-tolerance")
            cooldown_was_active = cd > 0
            if judged_toxic:
                penalty += pts
            if (e["judge"]["category"] == "zero-tolerance" and judged_toxic) or (
                judged_toxic and penalty >= thresh and not cooldown_was_active
            ):
                fires += 1
                penalty = 0
                cd = cooldown
            elif cooldown_was_active and judged_toxic:
                cd -= 1
        results[thresh] = fires
    return results


def main() -> None:
    classifier = LocalClassifier(CLASSIFIER_PATH)
    judge = LLMReasoner(model=JUDGE_MODEL)

    all_summaries: dict[str, Any] = {}

    for path_str in PILOT_FILES:
        path = Path(path_str)
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        rows = load_jsonl(path)
        if not rows:
            continue
        fmt = detect_format(rows[0])
        root_ctx = root_context_for(rows, fmt)

        out_path = OUT_DIR / f"{path.stem}_full_eval.jsonl"
        entries: list[dict[str, Any]] = []
        prev = ""
        print(f"\n=== {path.name} ({fmt}) ===")
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

        # Summaries
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

    out_summary = OUT_DIR / "pilots_full_eval_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n\nCross-transcript summary -> {out_summary}")


if __name__ == "__main__":
    main()
