import base64
import io

from PIL import Image

max_side = 1024
allowed_modes = {"RGB", "RGBA", "L"}


def preprocess(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}") from e
    if img.mode not in allowed_modes:
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode == "L":
        img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")
