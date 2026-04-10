from collections.abc import Iterator

from interfaces import Comment


def flatten_comments(
    comments: list[Comment], parent_id: str
) -> Iterator[tuple[Comment, str]]:
    """Recursively yields (comment, parent_id) from a nested comment tree."""
    for c in comments:
        yield c, parent_id
        if c.get("replies"):
            yield from flatten_comments(c["replies"], c["id"])
