import os
import logging
import asyncio
from typing import Any
from rag.retriever import retrieve
from rag.generator import generate
from vision.captioner import caption_image
from utils.cache import get_cached, set_cached
from utils.history import get_history

top_k = int(os.environ.get("TOP_K", 3))

logger = logging.getLogger(__name__)


async def route_text(query: str, user_id: int) -> dict[str, Any]:
    cached = get_cached("text", query)
    if cached:
        return cached

    history = get_history(user_id)
    chunks = retrieve(query, top_k)
    result = generate(query, chunks, history)
    set_cached("text", query, result)
    return result


async def route_image(image_bytes: bytes, user_id: int, user_hint: str | None = None) -> dict[str, Any]:
    cached = get_cached("image", image_bytes)
    if cached:
        return cached
    result = await asyncio.to_thread(caption_image, image_bytes, user_hint)
    set_cached("image", image_bytes, result)
    return result


async def route_summarize(history: list[dict]) -> str:
    from rag.generator import summarize_history
    return await asyncio.to_thread(summarize_history, history)
