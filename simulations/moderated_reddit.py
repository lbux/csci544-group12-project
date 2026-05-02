import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.agents import RedditDebateAgent
from core.llm_client import get_client
from core.moderation import (
    ActiveModerator,
    ModerationIntervener,
    ModerationReasoner,
    ToxicityClassifier,
)
from core.reddit_utils import (
    build_alignment_profiles,
    build_seed_history,
    load_submissions,
    save_history,
    select_seed_path,
    select_submission,
    thread_context_for,
)


def run_moderated_simulation(
    input_path: str | Path,
    model: str = "llama3.1:8b",
    judge_model: str = "llama3.2:3b",
    rounds: int = 3,
    out_dir: str | Path = "sim_debate_records",
    toxicity_threshold: float = 0.75,
    submission_index: int = 0,
    base_url: str = "http://localhost:11434/v1/",
    api_key: str = "ollama",
) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    """Continues a Reddit thread with aligned agents monitored by an Active Moderator."""
    topic = "abortion rights"

    submissions = load_submissions(input_path)
    submission = select_submission(submissions, submission_index, None)
    seed_path = select_seed_path(submission, min_seed_words=8)
    alignment_profiles = build_alignment_profiles(seed_path)
    history = build_seed_history(submission, seed_path)
    root_context = thread_context_for(submission)

    print(f"Topic: {topic} | Debaters: {model} | Judge: {judge_model}")
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

    moderator = ActiveModerator(
        reasoner=ModerationReasoner(client, judge_model, thinking=True),
        intervener=ModerationIntervener(client, judge_model, thinking=True),
        classifier=ToxicityClassifier(),
        toxicity_threshold=toxicity_threshold,
    )

    debate_id = uuid4().hex[:5]

    for round_idx in range(1, rounds + 1):
        print(f"Generated Round {round_idx} (ID: {debate_id})")
        print("-" * 50)

        for agent in agents:
            parent_text = history[-1].get("body", "") if history else ""

            text = agent.speak(history, submission)
            generated_turn = {  # pyright: ignore[reportUnknownVariableType]
                "id": f"generated_{uuid4().hex[:8]}",
                "author": agent.name,
                "body": text,
                "created_utc": time.time(),
                "replies": [],
                "round_idx": round_idx,
                "generated": True,
                "type": "comment",
            }
            history.append(generated_turn)  # pyright: ignore[reportUnknownArgumentType]

            intervention = moderator.observe(generated_turn, parent_text, root_context)  # pyright: ignore[reportUnknownArgumentType]
            if intervention:
                history.append(intervention)

            print("-" * 50)

    save_history(  # pyright: ignore[reportUnusedCallResult]
        debate_id=debate_id,
        history=history,
        submission=submission,
        topic=topic,
        out_dir=str(out_dir),
        prefix="moderated_reddit",
        model=model,
        judge_model=judge_model,
    )
    return history
