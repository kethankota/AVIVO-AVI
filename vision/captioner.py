import json
import logging
import os

import requests
from pydantic import BaseModel, Field, field_validator

from vision.preprocessor import preprocess

logger = logging.getLogger(__name__)

base_caption_prompt = """
    "Describe this image. Respond ONLY with a JSON object — no prose, no markdown. "
    "Use exactly this structure:\n"
    '{"caption": "<two-sentence description>", "tags": ["tag1", "tag2", "tag3"]}'"""

ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
vision_model = os.environ.get("VISION_MODEL", "llava:7b")


def build_prompt(user_hint: str | None) -> str:
    if user_hint:
        return (
            f'The user says: "{user_hint}". '
            f"With that in consideration, {base_caption_prompt}"
        )
    return base_caption_prompt


class ImageCaption(BaseModel):
    caption: str = Field(...,
                         description="Two-sentence description of the image.")
    tags: list[str] = Field(...,
                            description="Exactly 3 lowercase keyword tags.")

    @field_validator("caption")
    @classmethod
    def caption_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("caption must not be empty")
        return v

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, v: list[str]) -> list[str]:
        tags = [t.strip().lower() for t in v if t.strip()][:3]
        if not tags:
            raise ValueError("at least one tag is required")
        return tags


def call_llava(b64_image: str, prompt: str) -> str:
    url = f"{ollama_host}/api/generate"
    payload = {
        "model": vision_model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 200},
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.RequestException as e:
        logger.error("LLaVA request failed: %s", e)
        raise RuntimeError(f"Vision model unavailable: {e}") from e


def parse_response(raw: str) -> ImageCaption:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("JSON decode failed — raw response: %s", raw)
        raise ValueError(f"Model did not return valid JSON: {e}") from e

    return ImageCaption.model_validate(data)


def caption_image(image_bytes: bytes, user_hint: str | None = None) -> dict:
    b64 = preprocess(image_bytes)
    prompt = build_prompt(user_hint)
    raw = call_llava(b64, prompt)
    result = parse_response(raw)
    return result.model_dump()
