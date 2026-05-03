import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.agents import RedditDebateAgent
from core.llm_client import get_client
from core.reddit_utils import (
    build_alignment_profiles,
    build_seed_history,
    load_submissions,
    save_history,
    select_seed_path,
    select_submission,
)


def run_reddit_simulation(
    input_path: str | Path,
    model: str = "llama3.1:8b",
    rounds: int = 3,
    out_dir: str | Path = "sim_debate_records",
    submission_index: int = 0,
    base_url: str = "http://localhost:11434/v1/",
    api_key: str = "ollama",
) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    """Continues a real Reddit thread with agents aligned to the original users."""
    topic = "abortion rights"

    submissions = load_submissions(input_path)
    submission = select_submission(submissions, submission_index, None)
    seed_path = select_seed_path(submission, min_seed_words=8)
    alignment_profiles = build_alignment_profiles(seed_path)
    history = build_seed_history(submission, seed_path)

    print(f"Topic: {topic} | Model: {model} | Rounds: {rounds}")
    print(f"Continuing thread: {submission['title']}")
    print("=" * 50)

    client = get_client(base_url, api_key)
    agents = [
        RedditDebateAgent(
            client=client,
            model=model,
            stream=True,
            thinking=True,
            topic=topic,
            name=p["name"],
            persona=p["persona"],
            aligned_author=p["author"],
            observed_comments=p["observed_comments"],
        )
        for p in alignment_profiles
    ]

    debate_id = uuid4().hex[:5]

    for round_idx in range(1, rounds + 1):
        print(f"Generated Round {round_idx} (ID: {debate_id})")
        print("-" * 50)
        for agent in agents:
            text = agent.speak(history, submission)
            history.append(
                {
                    "id": f"generated_{uuid4().hex[:8]}",
                    "author": agent.name,
                    "body": text,
                    "created_utc": time.time(),
                    "replies": [],
                    "round_idx": round_idx,
                    "generated": True,
                    "type": "comment",
                }
            )
            print("-" * 50)

    # 4. Save
    save_history(
        debate_id,
        history,
        submission,
        topic,
        str(out_dir),
        prefix="reddit",
        model=model,
    )  # pyright: ignore[reportUnusedCallResult]
    return history
