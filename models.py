"""All OpenAI traffic lives here.

Two reasons to keep it in one file: the API key never leaves the server, and
MOCK mode can stand in for every call so the UI is clickable before you spend
anything.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import budget, config


class ModelError(RuntimeError):
    pass


def _client():
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ModelError("OPENAI_API_KEY is not set. Set it, or run with MOCK=1.")
    return OpenAI(api_key=key)


def live() -> bool:
    return not config.MOCK and bool(os.getenv("OPENAI_API_KEY"))


# --- Structured analysis -----------------------------------------------------

BRIEF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["product_name", "category", "materials", "dominant_colours",
                 "surface_pattern", "must_preserve", "scene_prompts", "risks"],
    "properties": {
        "product_name": {"type": "string"},
        "category": {"type": "string"},
        "materials": {"type": "array", "items": {"type": "string"}},
        "dominant_colours": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "hex"],
                "properties": {"name": {"type": "string"}, "hex": {"type": "string"}},
            },
        },
        "surface_pattern": {"type": "string"},
        "must_preserve": {"type": "array", "items": {"type": "string"}},
        "scene_prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "prompt"],
                "properties": {"label": {"type": "string"}, "prompt": {"type": "string"}},
            },
        },
        "risks": {"type": "array", "items": {"type": "string"}},
    },
}

INVENT_SYSTEM = """You brief a product photography pipeline for e-commerce listings.

There is no source image. The seller has described what they want and you are
writing the brief that will be generated from scratch.

Rules:
- Take the seller's description literally. Do not upgrade their product into
  something more premium, and do not add features they did not mention.
- dominant_colours: the colours you are specifying, as hex. If the seller named
  a colour, match it; otherwise choose and commit, because the generator needs a
  decision rather than a range.
- must_preserve: the details that must survive later refinement passes.
- scene_prompts: three complete photographs. Unlike the edit path, these DO
  describe the product, because nothing exists yet. Each should read like a
  brief to a photographer: the product and its material and finish, then the
  surface, backdrop, light direction and quality, colour temperature, shadow
  behaviour, lens and framing. Vary the three meaningfully - not three angles of
  one idea, but three different treatments a buyer would react to differently.
- No text, no logos, no brand marks, no packaging copy, no human hands or faces.
- risks: ways this specific product tends to be rendered wrong.
"""

ANALYST_SYSTEM = """You brief a product photography pipeline for e-commerce listings.

You receive renders or photographs of one product. Produce a factual brief.

Rules:
- Describe only what is visible. Never invent a brand, model number or material.
- dominant_colours must be sampled from the image, as hex, most prominent first.
- must_preserve lists the details an editing model must not alter: geometry,
  proportions, label text verbatim, logo placement, surface pattern, finish.
- scene_prompts: three distinct background and lighting treatments suited to the
  product's category and price point. Each prompt describes the SCENE ONLY -
  surface, backdrop, light direction, quality and colour temperature, shadow
  behaviour, camera framing. Never describe the product itself; it is composited
  in unchanged. No text, no logos, no props that imply a brand.
- risks: ways this specific product is likely to be rendered wrong.
"""

QA_SYSTEM = """You are the final visual check on a product listing image.

Colour and geometry have already been measured numerically and the product
region was composited back from the source, so do not comment on colour accuracy
or on whether the product changed shape.

Judge only what numbers cannot: does the lighting on the product match the
lighting of the background; do the contact shadows sit correctly; does the
product look placed in the scene or pasted onto it; is there any text, logo,
watermark or duplicated object that should not be there; would this pass as a
professional product photograph.

Reply as JSON: {"usable": true|false, "issues": [string], "fix_instruction": string}
fix_instruction is a single edit instruction to correct the worst issue, or "".
"""


# Vision inputs are counted in patches, and the limit is 30,000. A modern
# phone photo blows past that on its own: a 4000x3000 shot counts around 55,000
# and the call is rejected outright. Nothing is gained by sending full
# resolution either - the model is reading what the product IS, not inspecting
# it - so everything headed for a vision call gets capped first.
VISION_MAX_EDGE = 896


def _for_vision(img: Image.Image, max_edge: int = None) -> Image.Image:
    max_edge = max_edge or VISION_MAX_EDGE
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    k = max_edge / max(w, h)
    return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)


def _is_too_large(exc: Exception) -> bool:
    text = str(exc).lower()
    return "patches" in text or "resize the image" in text


def _b64_image(img: Image.Image, fmt: str = "PNG", max_edge: int = None) -> str:
    img = _for_vision(img, max_edge)
    buf = io.BytesIO()
    # Transparency carries no meaning to the vision model and PNG of a photo is
    # enormous, so flatten onto white and send JPEG.
    if fmt == "PNG" and img.mode == "RGBA":
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(canvas, img)
        fmt = "JPEG"
    img = img.convert("RGB") if fmt == "JPEG" else img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 88} if fmt == "JPEG" else {}))
    return base64.b64encode(buf.getvalue()).decode()


def _chat_json(system: str, user_parts: list, schema: dict | None, max_tokens: int = 4000) -> dict:
    client = _client()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_parts}]

    kwargs = {
        "model": config.REASONING_MODEL,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "brief", "strict": True, "schema": schema},
        }
    else:
        kwargs["response_format"] = {"type": "json_object"}

    budget.check_text()
    try:
        kwargs["reasoning_effort"] = config.REASONING_EFFORT
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("reasoning_effort", None)
        resp = client.chat.completions.create(**kwargs)
    budget.record("text", config.REASONING_MODEL)

    text = resp.choices[0].message.content or "{}"
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelError(f"{config.REASONING_MODEL} returned unparseable JSON: {text[:200]}") from exc


def analyse(images: list[Image.Image], user_note: str = "") -> dict:
    """Astra reads the product and writes the brief that drives every later call."""
    if not live():
        return _mock_brief(images)

    # Belt and braces. The cap above should make this impossible, but a
    # rejected call costs the whole run, so if the API still says the image is
    # too big we halve and try again rather than failing the job.
    last = None
    for edge in (VISION_MAX_EDGE, 640, 448):
        parts = [{"type": "text",
                  "text": f"Brief this product for marketplace listing photography.\n"
                          f"Seller note: {user_note or '(none)'}"}]
        for img in images[:6]:
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_b64_image(img, 'JPEG', edge)}"
                },
            })
        try:
            return _chat_json(ANALYST_SYSTEM, parts, BRIEF_SCHEMA)
        except Exception as exc:
            last = exc
            if not _is_too_large(exc):
                raise
    raise last


def invent(description: str) -> dict:
    """Astra turns a sentence from the seller into a full generation brief."""
    if not live():
        return _mock_invented_brief(description)

    parts = [{"type": "text",
              "text": f"The seller wants images of this product:\n\n{description}"}]
    return _chat_json(INVENT_SYSTEM, parts, BRIEF_SCHEMA)


def generate_scratch(prompts: list[str],
                     size: tuple[int, int] = None) -> list[tuple[str, Image.Image]]:
    """Text to image, no source. Flare, one call per scene."""
    size = size or config.DRAFT_SIZE
    if not live():
        return [(p, _mock_invented_scene(p, size)) for p in prompts]

    from imaging import size_string
    client = _client()
    budget.check_images(len(prompts))

    out = []
    for prompt in prompts:
        result = client.images.generate(
            model=config.DRAFT_MODEL,
            prompt=f"{prompt}\n\nCommercial product photograph. No text, no logos, "
                   f"no watermarks, no visible brand marks, no hands or people.",
            size=size_string(*size),
            quality=config.DRAFT_QUALITY,
            output_format="png",
        )
        budget.record("image", config.DRAFT_MODEL)
        out.append((prompt, _decode(result.data[0])))
    return out


def qa_review(final: Image.Image, brief: dict) -> dict:
    """The subjective half of the quality check."""
    if not live():
        return {"usable": True, "issues": [], "fix_instruction": ""}

    parts = [
        {"type": "text",
         "text": "Product: " + json.dumps({
             "name": brief.get("product_name"),
             "category": brief.get("category"),
             "must_preserve": brief.get("must_preserve", []),
         })},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{_b64_image(final, 'JPEG')}"}},
    ]
    out = _chat_json(QA_SYSTEM, parts, None, max_tokens=1500)
    out.setdefault("usable", True)
    out.setdefault("issues", [])
    out.setdefault("fix_instruction", "")
    return out


# --- Image generation and editing -------------------------------------------

PRESERVE_CLAUSE = (
    "Change the background and lighting environment only. Do not alter the product: "
    "keep its geometry, proportions, camera angle, surface pattern, finish and every "
    "label exactly as supplied. No added text, no logos, no watermarks, no borders, "
    "no duplicated objects, no props resting on or overlapping the product."
)


def _decode(item) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(item.b64_json))).convert("RGBA")


def generate_drafts(source: Image.Image, prompts: list[str],
                    size: tuple[int, int] = None) -> list[tuple[str, Image.Image]]:
    """Flare, one call per scene prompt. Fast and cheap enough to throw away."""
    size = size or config.DRAFT_SIZE
    if not live():
        return [(p, _mock_scene(source, p, size)) for p in prompts]

    from imaging import size_string
    client = _client()
    out = []

    buf = io.BytesIO()
    source.convert("RGBA").save(buf, format="PNG")

    budget.check_images(len(prompts))

    for prompt in prompts:
        buf.seek(0)
        result = client.images.edit(
            model=config.DRAFT_MODEL,
            image=[("source.png", buf.getvalue(), "image/png")],
            prompt=f"{prompt}\n\n{PRESERVE_CLAUSE}",
            size=size_string(*size),
            quality=config.DRAFT_QUALITY,
            output_format="png",
        )
        budget.record("image", config.DRAFT_MODEL)
        out.append((prompt, _decode(result.data[0])))
    return out


def final_edit(source: Image.Image, prompt: str, mask_png: bytes | None,
               size: tuple[int, int] = None) -> Image.Image:
    """Sunburst with the mask. This is the one that has to hold up."""
    size = size or config.FINAL_SIZE
    if not live():
        return _mock_scene(source, prompt, size, refine=True)

    from imaging import size_string
    client = _client()

    buf = io.BytesIO()
    source.convert("RGBA").resize(size, Image.LANCZOS).save(buf, format="PNG")

    kwargs = dict(
        model=config.FINAL_MODEL,
        image=[("source.png", buf.getvalue(), "image/png")],
        prompt=f"{prompt}\n\n{PRESERVE_CLAUSE}",
        size=size_string(*size),
        quality=config.FINAL_QUALITY,
        output_format="png",
    )
    if mask_png:
        kwargs["mask"] = ("mask.png", mask_png, "image/png")

    budget.check_images(1)
    result = client.images.edit(**kwargs)
    budget.record("image", config.FINAL_MODEL)
    return _decode(result.data[0])


# --- Mock mode ---------------------------------------------------------------

def _mock_brief(images: list[Image.Image]) -> dict:
    swatches = []
    if images:
        arr = np.asarray(images[0].convert("RGBA"))
        opaque = arr[arr[..., 3] > 0][:, :3] if (arr[..., 3] > 0).any() else arr[..., :3].reshape(-1, 3)
        if len(opaque):
            for q in (20, 55, 85):
                c = np.percentile(opaque, q, axis=0).astype(int)
                swatches.append({"name": f"tone {q}", "hex": "#%02x%02x%02x" % tuple(c)})

    return {
        "product_name": "Unidentified product (mock mode)",
        "category": "general merchandise",
        "materials": ["unknown"],
        "dominant_colours": swatches or [{"name": "neutral", "hex": "#b0b2b6"}],
        "surface_pattern": "not analysed in mock mode",
        "must_preserve": ["geometry", "proportions", "label text", "surface finish"],
        "scene_prompts": [
            {"label": "Studio sweep",
             "prompt": "Seamless light grey studio sweep, soft overhead key from the "
                       "upper left, gentle gradient falloff, single soft contact shadow."},
            {"label": "Warm oak",
             "prompt": "Oiled oak tabletop, late afternoon window light raking from the "
                       "right, long soft shadow, warm neutral backdrop thrown out of focus."},
            {"label": "Editorial concrete",
             "prompt": "Polished concrete plinth against a deep charcoal wall, hard "
                       "directional key from above, crisp shadow, cool daylight balance."},
        ],
        "risks": ["Mock mode: no model has seen this product."],
    }


def _mock_invented_brief(description: str) -> dict:
    words = description.strip().rstrip(".")
    return {
        "product_name": (words[:60] or "Unnamed product") + " (mock mode)",
        "category": "general merchandise",
        "materials": ["as described"],
        "dominant_colours": [{"name": "placeholder", "hex": "#8a8f98"}],
        "surface_pattern": "not analysed in mock mode",
        "must_preserve": ["silhouette", "colour", "material finish"],
        "scene_prompts": [
            {"label": "Studio sweep",
             "prompt": f"{words}, on a seamless light grey studio sweep, soft "
                       "overhead key from the upper left, single soft contact shadow."},
            {"label": "Warm oak",
             "prompt": f"{words}, on an oiled oak tabletop, late afternoon window "
                       "light raking from the right, long soft shadow."},
            {"label": "Editorial concrete",
             "prompt": f"{words}, on a polished concrete plinth against a deep "
                       "charcoal wall, hard directional key, crisp shadow."},
        ],
        "risks": ["Mock mode: nothing has been generated from this description."],
    }


def _mock_invented_scene(prompt: str, size: tuple[int, int]) -> Image.Image:
    """A placeholder frame so the scratch path is clickable offline."""
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    w, h = size

    top = tuple(rng.randint(60, 210) for _ in range(3))
    bottom = tuple(max(0, c - rng.randint(40, 100)) for c in top)
    grad = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (t, b) in enumerate(zip(top, bottom)):
        grad[..., i] = np.linspace(t, b, h, dtype=np.uint8)[:, None]
    img = Image.fromarray(grad, "RGB").convert("RGBA")

    # A simple object so the frame reads as a product shot rather than a gradient.
    body = tuple(rng.randint(40, 200) for _ in range(3))
    d = ImageDraw.Draw(img)
    bw, bh = int(w * 0.34), int(h * 0.42)
    x, y = (w - bw) // 2, int(h * 0.34)

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(
        [x - bw * 0.1, y + bh * 0.94, x + bw * 1.1, y + bh * 1.12], fill=(0, 0, 0, 100))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(bw * 0.04)))

    d = ImageDraw.Draw(img)
    d.rounded_rectangle([x, y, x + bw, y + bh], radius=int(bw * 0.12),
                        fill=(*body, 255))
    d.rounded_rectangle([x + bw * 0.08, y + bh * 0.06, x + bw * 0.42, y + bh * 0.5],
                        radius=int(bw * 0.06),
                        fill=tuple(min(255, c + 45) for c in body) + (90,))
    return img


def _mock_scene(source: Image.Image, prompt: str, size: tuple[int, int],
                refine: bool = False) -> Image.Image:
    """Synthesise a plausible background so the pipeline runs end to end offline."""
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    w, h = size
    top = tuple(rng.randint(40, 230) for _ in range(3))
    bottom = tuple(max(0, c - rng.randint(30, 90)) for c in top)

    grad = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (t, b) in enumerate(zip(top, bottom)):
        grad[..., i] = np.linspace(t, b, h, dtype=np.uint8)[:, None]
    bg = Image.fromarray(grad, "RGB").convert("RGBA")

    src = source.convert("RGBA")
    k = min(w / src.width, h / src.height) * (0.72 if not refine else 0.78)
    src = src.resize((max(1, int(src.width * k)), max(1, int(src.height * k))), Image.LANCZOS)
    x, y = (w - src.width) // 2, int((h - src.height) * 0.56)

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(
        [x + src.width * 0.08, y + src.height * 0.92,
         x + src.width * 0.92, y + src.height * 1.06],
        fill=(0, 0, 0, 90),
    )
    bg = Image.alpha_composite(bg, shadow.filter(ImageFilter.GaussianBlur(src.width * 0.03)))
    bg.paste(src, (x, y), src)
    return bg
