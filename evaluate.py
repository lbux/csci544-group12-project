"""Two-tier evaluation pipeline for debate transcripts.

Scores each LLM-generated turn with:
  Tier 1: local fine-tuned CGA classifier (toxicity probability)
  Tier 2: LLM-as-judge via Ollama (category, issue_type, points, explanation)

Handles two transcript formats produced by sim/:
  naive  — flat turn list from sim/naive_abortion_debate.py
           fields: debate_id, turn_idx, round_idx, agent, text
  reddit — mixed real + generated comments from sim/reddit_abortion_debate.py
           fields: debate_id, submission_id, title, turn_idx, round_idx,
                   generated, id, author, body, created_utc, replies, toxicity
           Real seed comments (generated=False) are not evaluated.
           Mediator turns (id starts with "mediator_") are skipped.

Writes <input_stem>_eval.jsonl with per-turn scores and prints aggregate metrics.

Usage:
    uv run python evaluate.py \\
        --input sim_debate_records/reddit_abortion_rights_1s3uc44_27a61_llama3.1_8b.jsonl \\
        --classifier cga_deberta_final \\
        --judge-model llama3.2:3b

Disable any tier with --classifier skip or --judge-model skip.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models import LLMReasoner


# -----------------------------------------------------------------------------
# Tier 1: local fine-tuned classifier
# -----------------------------------------------------------------------------


class LocalClassifier:
    """Fine-tuned CGA classifier loaded from a local checkpoint folder."""

    def __init__(self, path: str | Path) -> None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Classifier checkpoint not found at {resolved.resolve()}. "
                "Pass --classifier <path> or --classifier skip."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForSequenceClassification.from_pretrained(resolved)
        self.model.eval()

    def predict(self, text: str) -> float:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0][1].item())


# -----------------------------------------------------------------------------
# Transcript iteration
# -----------------------------------------------------------------------------


def detect_format(row: dict[str, Any]) -> str:
    if "body" in row and "author" in row and "generated" in row:
        return "reddit"
    if "text" in row and "agent" in row:
        return "naive"
    raise ValueError(f"Unrecognized transcript row shape: {sorted(row.keys())}")


def iter_evaluable_turns(
    rows: list[dict[str, Any]], fmt: str
) -> Iterator[tuple[dict[str, Any], str, str, str]]:
    """Yield (row, text, author, parent_text) for turns to evaluate.

    parent_text is the body of the immediately prior turn in the transcript,
    whether or not it was itself evaluable — the judge needs local context.
    """
    prev_body = ""
    for row in rows:
        if fmt == "naive":
            text = row.get("text", "") or ""
            author = row.get("agent", "") or ""
            yield row, text, author, prev_body
            prev_body = text
        else:
            body = row.get("body", "") or ""
            row_id = str(row.get("id", "") or "")
            is_mediator = row_id.startswith("mediator_")
            is_generated = bool(row.get("generated", False))
            if is_generated and not is_mediator:
                yield row, body, row.get("author", "") or "", prev_body
            prev_body = body


def root_context_for(rows: list[dict[str, Any]], fmt: str) -> str:
    if fmt == "reddit" and rows:
        title = (rows[0].get("title") or "").strip()
        body = (rows[0].get("body") or "").strip()
        combined = f"{title}\n\n{body}".strip() if body else title
        return combined
    return "Fixed-turn one-on-one LLM debate."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
    return rows


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def evaluate_file(
    input_path: Path,
    out_path: Path,
    classifier: LocalClassifier | None,
    judge: LLMReasoner | None,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    if not rows:
        raise ValueError(f"Empty transcript: {input_path}")
    fmt = detect_format(rows[0])
    root_ctx = root_context_for(rows, fmt)

    eval_entries: list[dict[str, Any]] = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for row, text, author, parent_text in iter_evaluable_turns(rows, fmt):
            entry: dict[str, Any] = {
                "source_file": input_path.name,
                "transcript_format": fmt,
                "debate_id": row.get("debate_id"),
                "turn_idx": row.get("turn_idx"),
                "round_idx": row.get("round_idx"),
                "author": author,
                "text": text,
            }

            if classifier is not None:
                entry["roberta_toxicity"] = classifier.predict(text)

            if judge is not None:
                result = judge.analyze_intent(
                    comment_body=text,
                    parent_body=parent_text,
                    thread_context=root_ctx,
                )
                entry["judge"] = {
                    "category": result.category,
                    "issue_type": result.issue_type,
                    "points": result.points,
                    "explanation": result.explanation,
                }

            eval_entries.append(entry)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()

    return aggregate(eval_entries)


def aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(entries)
    summary: dict[str, Any] = {"n_turns": n}
    if n == 0:
        return summary

    rob_scores = [e["roberta_toxicity"] for e in entries if "roberta_toxicity" in e]
    if rob_scores:
        summary["roberta_mean"] = sum(rob_scores) / len(rob_scores)
        summary["roberta_max"] = max(rob_scores)
        summary["roberta_frac_over_0_6"] = sum(
            1 for s in rob_scores if s >= 0.6
        ) / len(rob_scores)

    judge_entries = [e["judge"] for e in entries if "judge" in e]
    if judge_entries:
        cats = [j["category"] for j in judge_entries]
        summary["judge_category_counts"] = {
            c: cats.count(c) for c in sorted(set(cats))
        }
        summary["judge_points_total"] = sum(j["points"] for j in judge_entries)
        summary["judge_points_mean"] = summary["judge_points_total"] / len(
            judge_entries
        )

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Transcript jsonl file(s) from sim_debate_records/.",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_records",
        help="Output directory for *_eval.jsonl.",
    )
    parser.add_argument(
        "--classifier",
        default="cga_deberta_final",
        help="Path to local fine-tuned classifier, or 'skip' to disable Tier 1.",
    )
    parser.add_argument(
        "--judge-model",
        default="llama3.2:3b",
        help="Ollama model id for LLM-as-judge, or 'skip' to disable Tier 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier = (
        None if args.classifier == "skip" else LocalClassifier(args.classifier)
    )
    judge = None if args.judge_model == "skip" else LLMReasoner(model=args.judge_model)

    all_summaries: dict[str, dict[str, Any]] = {}
    for input_str in args.input:
        input_path = Path(input_str)
        out_path = out_dir / (input_path.stem + "_eval.jsonl")
        print(f"\n=== {input_path.name} ===")
        summary = evaluate_file(
            input_path=input_path,
            out_path=out_path,
            classifier=classifier,
            judge=judge,
        )
        print(f"-> wrote {out_path}")
        print(json.dumps(summary, indent=2))
        all_summaries[input_path.name] = summary

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll summaries written to {summary_path}")


if __name__ == "__main__":
    main()
