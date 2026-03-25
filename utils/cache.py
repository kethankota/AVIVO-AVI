import hashlib
from typing import Any

text_cache: dict[str, Any] = {}
image_cache: dict[str, Any] = {}


def text_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def image_key(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def get_cached(kind: str, payload) -> Any | None:
    if kind == "text":
        return text_cache.get(text_key(payload))
    if kind == "image":
        return image_cache.get(image_key(payload))
    return None


def set_cached(kind: str, payload, value: Any) -> None:
    if kind == "text":
        text_cache[text_key(payload)] = value
    elif kind == "image":
        image_cache[image_key(payload)] = value


def cache_stats() -> dict:
    return {
        "text_entries": len(text_cache),
        "image_entries": len(image_cache),
    }
