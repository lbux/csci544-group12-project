# pyright: reportExplicitAny=false, reportAny=false
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schemas import AlignmentProfile, Comment, RedditThread


def load_submissions(path: str | Path) -> list[RedditThread]:
    submissions: list[RedditThread] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    submissions.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} in {path}"
                    ) from exc
    if not submissions:
        raise ValueError(f"No submissions found in {path}")
    return submissions


def select_submission(
    submissions: list[RedditThread], submission_index: int, submission_id: str | None
) -> RedditThread:
    if submission_id is not None:
        for submission in submissions:
            if submission["submission_id"] == submission_id:
                return submission
        raise ValueError(f"Could not find submission_id={submission_id}")
    if submission_index < 0 or submission_index >= len(submissions):
        raise ValueError(
            f"submission-index must be between 0 and {len(submissions) - 1}"
        )
    return submissions[submission_index]


def is_usable_comment(comment: Comment, min_words: int = 1) -> bool:
    author = comment["author"].strip()
    body = comment["body"].strip()
    if author.lower() == "automoderator" or body.lower() in {"[deleted]", "[removed]"}:
        return False
    return len(body.split()) >= min_words


def iter_comment_paths(
    comments: list[Comment], min_words: int, path: list[Comment] | None = None
) -> Iterator[list[Comment]]:
    current_path = path or []
    for comment in comments:
        if not is_usable_comment(comment, min_words=min_words):
            yield from iter_comment_paths(comment["replies"], min_words, current_path)
            continue
        next_path = current_path + [comment]
        yield next_path
        yield from iter_comment_paths(comment["replies"], min_words, next_path)


def select_seed_path(submission: RedditThread, min_seed_words: int) -> list[Comment]:
    paths = list(iter_comment_paths(submission["comments"], min_seed_words))
    if not paths:
        raise ValueError("No usable Reddit comment chain found.")

    def score(path: list[Comment]) -> tuple[float, float, float, int]:
        toxicity_scores = [float(c.get("toxicity", 0.0)) for c in path]
        return (
            max(toxicity_scores, default=0.0),
            sum(toxicity_scores) / (len(toxicity_scores) or 1),
            sum(toxicity_scores),
            len(path),
        )

    return max(paths, key=score)


def select_alignment_authors(seed_path: list[Comment]) -> tuple[str, str]:
    authors: list[str] = []
    for comment in reversed(seed_path):
        if not is_usable_comment(comment):
            continue
        author = comment["author"].strip()
        if author not in authors:
            authors.append(author)
        if len(authors) == 2:
            return authors[1], authors[0]
    raise ValueError(
        "Need at least two distinct usable Reddit authors in the seed chain."
    )


def build_alignment_profiles(
    seed_path: list[Comment],
) -> tuple[AlignmentProfile, AlignmentProfile]:
    author_1, author_2 = select_alignment_authors(seed_path)

    def profile_for(author: str, index: int) -> AlignmentProfile:
        observed: list[Comment] = [
            {
                "id": c["id"],
                "author": c["author"],
                "body": c["body"],
                "created_utc": c["created_utc"],
                "replies": [],
            }
            for c in seed_path
            if c["author"].strip() == author
        ]
        return {
            "author": author,
            "name": f"Aligned User {index} (u/{author})",
            "persona": "Use only this user's observed comments in the seed chain as evidence for stance, arguments, priorities, and tone.",
            "observed_comments": observed,
        }

    return profile_for(author_1, 1), profile_for(author_2, 2)


def thread_context_for(submission: RedditThread) -> str:
    post_body = (submission.get("selftext") or "").strip()
    return (
        f"Title: {submission['title']}\nPost: {post_body}"
        if post_body
        else f"Title: {submission['title']}"
    )


def build_seed_history(
    submission: RedditThread, seed_path: list[Comment]
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [
        {
            "id": submission["submission_id"],
            "author": submission["author"],
            "body": thread_context_for(submission),
            "created_utc": submission["created_utc"],
            "replies": [],
            "type": "post",
            "generated": False,
        }
    ]
    for c in seed_path:
        history.append(
            {
                "id": c["id"],
                "author": c["author"],
                "body": c["body"].strip(),
                "created_utc": c["created_utc"],
                "replies": [],
                "type": "comment",
                "generated": False,
                "toxicity": float(c.get("toxicity", 0.0)),
            }
        )
    return history


def safe_filename_piece(value: str | None) -> str:
    if not value:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def save_history(
    debate_id: str,
    history: list[dict[str, Any]],
    submission: RedditThread,
    topic: str,
    out_dir: str,
    prefix: str = "moderated_reddit",
    model: str | None = None,
    **kwargs: Any,
) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_suffix = f"_{safe_filename_piece(model)}" if model else ""
    topic_name = safe_filename_piece(topic)
    sub_id = safe_filename_piece(submission["submission_id"])

    out_path = (
        Path(out_dir)
        / f"{prefix}_{topic_name}_{sub_id}_{debate_id}{model_suffix}.jsonl"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for turn_idx, turn in enumerate(history):
            record = {
                "debate_id": debate_id,
                "submission_id": submission["submission_id"],
                "submission_url": submission.get("submission_url"),
                "title": submission["title"],
                "turn_idx": turn_idx,
                **kwargs,
                **turn,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")  # pyright: ignore[reportUnusedCallResult]
    print(f"History saved to {out_path}")
    return out_path
