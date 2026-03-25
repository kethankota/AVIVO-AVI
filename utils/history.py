import os

from collections import defaultdict, deque
from typing import Any

max_history = int(os.environ.get("MAX_HISTORY", 3))

store: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))


def add_to_history(user_id: int, kind: str, user_input: str, bot_output: str) -> None:
    store[user_id].append({
        "type": kind,
        "input": user_input,
        "output": bot_output,
    })


def get_history(user_id: int) -> list[dict[str, Any]]:
    return list(store[user_id])


def clear_history(user_id: int) -> None:
    store[user_id].clear()
