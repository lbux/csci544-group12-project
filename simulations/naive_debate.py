import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.agents import NaiveDebateAgent
from core.llm_client import get_client
from core.reddit_utils import safe_filename_piece


def save_naive_history(
    debate_id: str,
    history: list[dict[str, Any]],  # pyright: ignore[reportExplicitAny]
    topic: str,
    model: str,
    out_dir: str | Path,
) -> None:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    topic_name = safe_filename_piece(topic)
    model_name = safe_filename_piece(model)
    out_path = out_dir_path / f"naive_{topic_name}_{debate_id}_{model_name}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for turn in history:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")  # pyright: ignore[reportUnusedCallResult]
    print(f"Naive debate history saved to {out_path}")


def run_naive_simulation(
    model: str = "llama3.1:8b",
    rounds: int = 3,
    out_dir: str | Path = "sim_debate_records",
    base_url: str = "http://localhost:11434/v1/",
    api_key: str = "ollama",
) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    """Runs a blank-slate debate simulation between two predefined personas."""
    client = get_client(base_url, api_key)
    topic = "abortion rights"

    agent_1 = NaiveDebateAgent(
        client=client,
        model=model,
        stream=True,
        thinking=True,
        topic=topic,
        name="Pro-Choice Advocate",
        persona="You support legal abortion access and emphasize bodily autonomy, privacy, and medical complexity.",
    )
    agent_2 = NaiveDebateAgent(
        client=client,
        model=model,
        stream=True,
        thinking=True,
        topic=topic,
        name="Pro-Life Advocate",
        persona="You oppose abortion and emphasize fetal life, moral responsibility, and legal protection for the unborn.",
    )

    print(f"Starting Naive Debate on '{topic}' using {model}")
    print("=" * 50)

    history: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]
    debate_id = uuid4().hex[:5]

    for round_idx in range(1, rounds + 1):
        print(f"Round {round_idx} (ID: {debate_id})")
        print("-" * 50)

        for agent in [agent_1, agent_2]:
            text = agent.speak(history)
            history.append(
                {
                    "debate_id": debate_id,
                    "round_idx": round_idx,
                    "agent": agent.name,
                    "text": text,
                }
            )
            print("-" * 50)

    save_naive_history(debate_id, history, topic, model, out_dir)
    return history
