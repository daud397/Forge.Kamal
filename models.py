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
- For textiles, surface_pattern is the most important field you write. It is what
  a later model uses to reproduce the cloth, so be specific enough to redraw
  from: name the motif, its approximate size relative to a pillowcase, how it
  repeats (scattered, half-drop, straight, directional), the spacing between
  motifs, the number of colours in the motif, whether the print is placed or
  all-over, and what the reverse side looks like if any of it is visible.
  "Floral print" is useless. "Small scattered floral sprigs about 4cm across,
  five per pillow width, two-colour on a pale sage ground, plain white reverse"
  is the standard.
- must_preserve lists the details an editing model must not alter: geometry,
  proportions, label text verbatim, logo placement, surface pattern, finish.
- The seller's note is an instruction, not background reading. If it names a
  setting, a colour, an angle, a mood or a constraint, every scene prompt must
  obey it. If it says the product's design must be matched exactly, say so in
  each prompt rather than describing an alternative.
- scene_prompts: distinct background and lighting treatments suited to the
  product's category and price point, and to whatever the note asked for. Each prompt describes the SCENE ONLY -
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


def generate_scratch(prompts: list[str], size: tuple[int, int] = None,
                     directive: str = "",
                     brief: dict | None = None) -> list[tuple[str, Image.Image]]:
    """Text to image, no source. Flare, one call per scene."""
    size = size or config.DRAFT_SIZE
    if not live():
        return [(p, _mock_invented_scene(p, size)) for p in prompts]

    from imaging import size_string
    client = _client()
    dims = size_string(*size)

    def one(prompt):
        result = client.images.generate(
            model=config.DRAFT_MODEL,
            prompt=f"{compose_prompt(prompt, directive, brief, scratch=True)}\n\n"
                   "Commercial product photograph. No text, no logos, no "
                   "watermarks, no visible brand marks, no hands or people.",
            size=dims,
            quality=config.DRAFT_QUALITY,
            output_format="png",
        )
        return prompt, _decode(result.data[0])

    return _in_parallel(one, prompts, config.DRAFT_MODEL)


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

def _as_files(images) -> list[tuple[str, bytes, str]]:
    """All reference images, in the order the seller attached them.

    Order is the contract: prompts written in this mill say "first image" and
    "second image", and that must mean attachment order, every time.
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
    files = []
    for i, img in enumerate(images):
        buf = io.BytesIO()
        img.convert("RGBA").save(buf, format="PNG")
        files.append((f"image_{i + 1}.png", buf.getvalue(), "image/png"))
    return files


def compose_prompt(scene: str, directive: str = "", brief: dict | None = None,
                   scratch: bool = False, n_images: int = 1) -> str:
    """Build what the image model actually receives.

    Three things belong in every generation prompt and only one of them used to
    be there:

    1. What the person asked for, in their own words. Paraphrasing a brief like
       "pick the exact same colour, motif and reversible design" into a scene
       sentence loses precisely the constraints they cared about.
    2. The facts read off the reference image - sampled hex colours, the motif,
       the repeat. The analysis step extracts these and they were going unused.
    3. The scene.

    Order matters. The strictest requirement goes first, because an instruction
    buried under scene description gets weighted like set dressing.
    """
    parts = []

    if n_images > 1:
        parts.append(f"{n_images} reference images are attached, in the order the "
                     "seller attached them. 'First image' or 'image 1' means the "
                     "first attachment, 'second image' or 'image 2' the second, "
                     "and so on.")

    if directive.strip():
        parts.append("WHAT IS REQUIRED, in the seller's own words:\n"
                     + directive.strip())

    # With several references the sampled colours and pattern are a blend of
    # all of them, and a transfer job needs the design read from one image, not
    # an average. The images speak for themselves; the directive says which is
    # which.
    if brief and n_images <= 1:
        facts = []
        colours = brief.get("dominant_colours") or []
        if colours:
            facts.append("Colours, to be matched exactly: " + ", ".join(
                f"{c.get('name', '')} {c.get('hex', '')}".strip() for c in colours[:5]))
        if brief.get("surface_pattern"):
            facts.append("Pattern and repeat: " + brief["surface_pattern"])
        if brief.get("materials"):
            facts.append("Material: " + ", ".join(brief["materials"][:3]))
        keep = brief.get("must_preserve") or []
        if keep:
            facts.append("Must not change: " + "; ".join(keep[:6]))
        if facts:
            label = ("THE PRODUCT TO RENDER:" if scratch
                     else "THE PRODUCT IN THE SUPPLIED IMAGE, to be reproduced exactly:")
            parts.append(label + "\n" + "\n".join(facts))

    parts.append("SCENE AND LIGHTING:\n" + scene)
    return "\n\n".join(parts)


# Subordinated to the seller's requirement, never in competition with it. The
# previous wording ordered the model to keep the product's surface pattern
# "exactly as supplied" on every call - including calls whose entire purpose
# was to change the pattern. The seller would ask for a design transfer and the
# clause would forbid it two lines later; the model obeyed the clause and the
# portal appeared to do nothing.
PRESERVE_CLAUSE = (
    "The requirement stated at the top takes precedence over everything else in "
    "this prompt. Follow it exactly. Whatever it does not ask to change, keep "
    "exactly as supplied: geometry, proportions, camera angle, surface pattern, "
    "finish, labels. If it asks only for a scene or background, do not alter the "
    "product at all. Never add text, logos, watermarks or borders, never "
    "duplicate objects, and never rest props on or over the product."
)


def _in_parallel(fn, prompts: list[str], model: str):
    """Run the image calls together and keep the requested order.

    These calls spend almost all their time waiting on the API, so generating
    three scenes one after another costs three times the latency for no reason.
    Budget is checked once up front, so a batch that would breach the cap is
    refused whole rather than halfway through.
    """
    from concurrent.futures import ThreadPoolExecutor

    budget.check_images(len(prompts))

    with ThreadPoolExecutor(max_workers=min(len(prompts), 4)) as pool:
        results = list(pool.map(fn, prompts))

    for _ in results:
        budget.record("image", model)
    return results


def _decode(item) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(item.b64_json))).convert("RGBA")


def generate_drafts(source, prompts: list[str],
                    size: tuple[int, int] = None, directive: str = "",
                    brief: dict | None = None) -> list[tuple[str, Image.Image]]:
    """source is one image or an ordered list; every one goes to the model."""
    """Flare, one call per scene prompt. Fast and cheap enough to throw away."""
    size = size or config.DRAFT_SIZE
    first = source[0] if isinstance(source, (list, tuple)) else source
    if not live():
        return [(p, _mock_scene(first, p, size)) for p in prompts]

    from imaging import size_string
    client = _client()

    files = _as_files(source)
    dims = size_string(*size)

    def one(prompt):
        result = _edit_with_fidelity(client, dict(
            model=config.DRAFT_MODEL,
            image=files,
            prompt=f"{compose_prompt(prompt, directive, brief, n_images=len(files))}"
                   f"\n\n{PRESERVE_CLAUSE}",
            size=dims,
            quality=config.DRAFT_QUALITY,
            output_format="png",
        ))
        return prompt, _decode(result.data[0])

    return _in_parallel(one, prompts, config.DRAFT_MODEL)


def transfer_generate(sources: list[Image.Image], prompt: str,
                      size: tuple[int, int] = None) -> Image.Image:
    """One image, final quality, from several references.

    This is the shape of the mill's daily job - design from one image, target
    from another - and it gets a single good result rather than a pair of cheap
    previews, because at draft quality a fine motif cannot survive and the
    options look wrong before anyone has picked one.
    """
    size = size or config.FINAL_SIZE
    if not live():
        return _mock_scene(sources[0], prompt, size, refine=True)

    from imaging import size_string
    client = _client()
    budget.check_images(1)

    kwargs = dict(
        model=config.FINAL_MODEL,
        image=_as_files(sources),
        prompt=prompt,
        size=size_string(*size),
        quality=config.FINAL_QUALITY,
        output_format="png",
    )
    result = _edit_with_fidelity(client, kwargs)
    budget.record("image", config.FINAL_MODEL)
    return _decode(result.data[0])


def _edit_with_fidelity(client, kwargs):
    """input_fidelity="high" tells the model to preserve fine detail from the
    input images - motif linework, exact colours - which is the whole point of
    a reference. Not every deployment accepts the parameter, so the call falls
    back to a plain edit rather than failing the run."""
    try:
        return client.images.edit(**kwargs, input_fidelity="high")
    except Exception as exc:
        if "input_fidelity" not in str(exc):
            raise
        return client.images.edit(**kwargs)


def final_edit(source, prompt: str, mask_png: bytes | None,
               size: tuple[int, int] = None,
               extra_refs: list[Image.Image] | None = None) -> Image.Image:
    """Sunburst with the mask. This is the one that has to hold up.

    The first image is the one being edited - a mask, if given, applies to it.
    extra_refs ride along so an instruction like "match the first image's
    design" still has the first image to look at during the final pass.
    """
    size = size or config.FINAL_SIZE
    if not live():
        return _mock_scene(source, prompt, size, refine=True)

    from imaging import size_string
    client = _client()

    buf = io.BytesIO()
    source.convert("RGBA").resize(size, Image.LANCZOS).save(buf, format="PNG")
    files = [("image_1.png", buf.getvalue(), "image/png")]
    for i, ref in enumerate(extra_refs or []):
        rb = io.BytesIO()
        ref.convert("RGBA").save(rb, format="PNG")
        files.append((f"image_{i + 2}.png", rb.getvalue(), "image/png"))

    kwargs = dict(
        model=config.FINAL_MODEL,
        image=files,
        prompt=f"{prompt}\n\n{PRESERVE_CLAUSE}",
        size=size_string(*size),
        quality=config.FINAL_QUALITY,
        output_format="png",
    )
    if mask_png:
        kwargs["mask"] = ("mask.png", mask_png, "image/png")

    budget.check_images(1)
    result = _edit_with_fidelity(client, kwargs)
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
