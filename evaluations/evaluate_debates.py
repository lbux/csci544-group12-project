import json
from pathlib import Path
from typing import Any

from core.llm_client import generate_structured_output, get_client
from core.schemas import DebateEvaluation


def load_history(filepath: str | Path) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    history: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line))  # pyright: ignore[reportAny]
    return history


def format_transcript(history: list[dict[str, Any]]) -> str:  # pyright: ignore[reportExplicitAny]
    """Converts the JSONL history into a readable script for the Judge LLM."""
    transcript: list[str] = []
    for turn in history:
        agent = turn.get("agent") or turn.get("author") or "Unknown"
        text = turn.get("text") or turn.get("body") or ""

        if turn.get("type") == "intervention":
            transcript.append(f"\n[MODERATOR INTERVENTION]: {text}\n")
        else:
            transcript.append(f"{agent}: {text}")

    return "\n".join(transcript)


def evaluate_transcript(
    input_file: str | Path,
    out_dir: str | Path = "eval_records",
    judge_model: str = "llama3.2:3b",
    base_url: str = "http://localhost:11434/v1/",
    api_key: str = "ollama",
) -> DebateEvaluation | None:
    """Runs an LLM judge over a finished debate and saves the score."""
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return None

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    history = load_history(input_path)
    transcript = format_transcript(history)

    client = get_client(base_url, api_key)

    system_prompt = (
        "You are an impartial, expert debate judge. Read the following debate transcript "
        "and evaluate it on alignment (staying in character), argument quality, and toxicity. "
        "You MUST declare a 'winner' by choosing exactly 'Agent 1', 'Agent 2', or 'Tie'. "
        "Do not output null."
    )

    user_prompt = f"Here is the transcript:\n\n{transcript}\n\nProvide your evaluation."

    print(f"Judging {input_path.name} using {judge_model}...")

    try:
        evaluation = generate_structured_output(
            client=client,
            model=judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=DebateEvaluation,
            thinking=False,
        )

        eval_filename = out_dir_path / f"eval_{input_path.name}"
        with open(eval_filename, "w", encoding="utf-8") as f:
            f.write(evaluation.model_dump_json(indent=2))  # pyright: ignore[reportUnusedCallResult]

        print(f"Evaluation saved to {eval_filename}")
        return evaluation

    except Exception as e:
        print(f"Evaluation failed for {input_path.name}: {e}")
        return None


def run_batch_evaluation(
    sim_dir: str | Path = "sim_debate_records",
    out_dir: str | Path = "eval_records",
    judge_model: str = "llama3.2:3b",
) -> None:
    """Evaluates all .jsonl files in the simulation directory."""
    sim_path = Path(sim_dir)
    for file in sim_path.glob("*.jsonl"):
        evaluate_transcript(file, out_dir, judge_model)  # pyright: ignore[reportUnusedCallResult]
