import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Union
from urllib.parse import urljoin, urlparse

import requests


# ---------- CONFIG ----------
USER_AGENT = "school-project-nlp:1.0 (by u/your_username)"

SUBREDDITS = [
    # "Abortiondebate",
    # "DebateReligion",
    # "PoliticalDebate",
    # "changemyview",
    # "gaming",
    "technology"
    # add more here
]

MAX_POSTS_PER_SUBREDDIT = 200
POST_PAGE_LIMIT = 100
SLEEP_BETWEEN_REQUESTS = 0.1
OUTPUT_DIR = "."  # change if you want

BASE_URL = "https://reddit.com"


# ---------- HTTP CLIENT ----------
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


# ---------- HELPERS ----------
def safe_body(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return text.replace("\u0000", "").strip()


def subreddit_output_path(subreddit_name: str) -> str:
    return os.path.join(OUTPUT_DIR, f"reddit_{subreddit_name.lower()}.jsonl")


def iter_jsonl_objects(filename: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(filename):
        return

    with open(filename, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {filename} on line {line_number}") from exc
            if isinstance(data, dict):
                yield data


def load_subreddit_posts(subreddit_name: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl_objects(subreddit_output_path(subreddit_name)))


def fetch_json(url: str, params: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], List[Any]]:
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def is_deleted_or_removed(body: Optional[str]) -> bool:
    return body in {"[deleted]", "[removed]"}


def extract_submission_id(post_url: str) -> Optional[str]:
    parts = [part for part in urlparse(post_url).path.split("/") if part]
    if "comments" in parts:
        idx = parts.index("comments")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def parse_comment_data(comment_data: Dict[str, Any], include_permalink: bool = True) -> Optional[Dict[str, Any]]:
    body = safe_body(comment_data.get("body"))
    if is_deleted_or_removed(body):
        return None

    replies: List[Dict[str, Any]] = []
    replies_data = comment_data.get("replies")
    if isinstance(replies_data, dict):
        reply_children = replies_data.get("data", {}).get("children", [])
    else:
        reply_children = []

    for child in reply_children:
        if child.get("kind") != "t1":
            continue
        reply_data = parse_comment_data(child.get("data", {}), include_permalink=False)
        if reply_data is not None:
            replies.append(reply_data)

    comment = {
        "id": comment_data.get("id"),
        "body": body,
        "author": safe_body(comment_data.get("author")),
        "created_utc": comment_data.get("created_utc"),
        "parent_id": comment_data.get("parent_id"),
        "replies": replies,
    }
    if include_permalink:
        comment["permalink"] = (
            urljoin(BASE_URL, comment_data["permalink"]) if comment_data.get("permalink") else None
        )
    return comment


def fetch_submission_urls(subreddit_name: str, limit_posts: int = 1000) -> List[str]:
    """
    Get up to limit_posts URLs from /new.json by paging Reddit's JSON feed.
    """
    subreddit_name = subreddit_name.strip().replace("r/", "")

    urls: List[str] = []
    after: Optional[str] = None

    while len(urls) < limit_posts:
        params: Dict[str, Any] = {"limit": POST_PAGE_LIMIT}
        if after:
            params["after"] = after

        payload = fetch_json(f"{BASE_URL}/r/{subreddit_name}/new.json", params=params)
        children = payload.get("data", {}).get("children", [])
        if not children:
            break

        page_added = 0
        for child in children:
            post_data = child.get("data", {})
            permalink = post_data.get("permalink")
            if not permalink:
                continue

            urls.append(urljoin(BASE_URL, permalink))
            page_added += 1

            if len(urls) >= limit_posts:
                break

        if page_added == 0:
            break

        after = payload.get("data", {}).get("after")
        if not after:
            break

        time.sleep(0.01)

    return urls


def fetch_submission_tree(submission_url: str) -> Dict[str, Any]:
    """
    Fetch one submission and recursively collect its comment tree.
    """
    payload = fetch_json(submission_url + ".json")
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Unexpected submission payload format for {submission_url}")

    submission_data = payload[0].get("data", {}).get("children", [{}])[0].get("data", {})
    comment_children = payload[1].get("data", {}).get("children", [])

    top_level_comments: List[Dict[str, Any]] = []
    for child in comment_children:
        if child.get("kind") != "t1":
            continue
        comment_data = parse_comment_data(child.get("data", {}))
        if comment_data is not None:
            top_level_comments.append(comment_data)

    return {
        "submission_id": submission_data.get("id") or extract_submission_id(submission_url),
        "submission_url": submission_url,
        "title": safe_body(submission_data.get("title")),
        "author": safe_body(submission_data.get("author")),
        "created_utc": submission_data.get("created_utc"),
        "comments": top_level_comments,
    }


def process_subreddit(subreddit_name: str) -> str:
    """
    Build one JSONL file for the whole subreddit, with one post per line.
    """
    subreddit_name = subreddit_name.strip().replace("r/", "")
    print(f"Processing r/{subreddit_name}...")

    submission_urls = fetch_submission_urls(subreddit_name, limit_posts=MAX_POSTS_PER_SUBREDDIT)
    existing_posts = load_subreddit_posts(subreddit_name)

    completed_urls = {
        post.get("submission_url")
        for post in existing_posts
        if isinstance(post, dict) and post.get("submission_url")
    }
    pending_urls = [url for url in submission_urls if url not in completed_urls]

    print(
        f"  Found {len(completed_urls)} saved posts, "
        f"{len(pending_urls)} remaining in current listing."
    )

    for i, url in enumerate(pending_urls, start=1):
        try:
            print(f"  [{i}/{len(pending_urls)}] {url}")
            post_data = fetch_submission_tree(url)
            append_post_jsonl(subreddit_name, post_data)
            completed_urls.add(url)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        except Exception as e:
            print(f"  Skipping {url}: {e}")

    return subreddit_output_path(subreddit_name)


def append_post_jsonl(subreddit_name: str, post: Dict[str, Any]) -> str:
    filename = subreddit_output_path(subreddit_name)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(post, ensure_ascii=False) + "\n")
    return filename


# ---------- MAIN ----------
def main():
    for subreddit_name in SUBREDDITS:
        filename = process_subreddit(subreddit_name)
        print(f"Saved {filename}")


if __name__ == "__main__":
    main()
