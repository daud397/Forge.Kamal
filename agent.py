"""Natural-language control over the pipeline.

Astra gets a description of where the job currently stands plus a set of tools
that map onto pipeline stages. It decides what to call; this module executes.

Every tool here is something the buttons can already do. The chat is a second
way in, not a second implementation - so a run driven by conversation and a run
driven by clicking end up in exactly the same state.
"""
from __future__ import annotations

import json
import re

import config, models, pipeline, store
from presets import PRESETS

MAX_HISTORY = 16   # turns kept in context; the job state carries the rest


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_scenes",
            "description": (
                "Replace the scene options and generate fresh drafts. Use when the "
                "person describes a background, setting, surface, mood or lighting "
                "they want, or asks for different options. Each prompt describes the "
                "BACKGROUND AND LIGHTING ONLY - never the product, which is "
                "composited in unchanged."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["scenes"],
                "properties": {
                    "scenes": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["label", "prompt"],
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "Two or three words for the card, e.g. 'Marble counter'.",
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": (
                                        "Full scene: surface, backdrop, light direction and "
                                        "quality, colour temperature, shadow behaviour, "
                                        "framing. Include the product itself only in "
                                        "scratch mode."
                                    ),
                                },
                            },
                        },
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "choose_draft",
            "description": (
                "Pick one draft and run the high-fidelity masked edit on it. Use when "
                "the person indicates which option they want - by number, by name, or "
                "by describing it ('the wooden one')."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index"],
                "properties": {
                    "index": {"type": "integer", "description": "Zero-based draft index."},
                    "instruction": {
                        "type": "string",
                        "description": "Optional extra direction for the final pass, e.g. 'shorten the shadow'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revise_final",
            "description": (
                "Re-run the final edit on the same draft with an additional "
                "instruction. Use when the person wants the finished image changed "
                "rather than a different scene."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["instruction"],
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "What to change, e.g. 'lower the key light and warm it slightly'.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_image",
            "description": (
                "Edit the image this run already has, directly, with no drafting "
                "round. Use when the person wants something added, removed or "
                "altered in the picture as it stands - props, furniture, a rug, "
                "a different surface. Nothing is masked on this path, so say so "
                "if the product itself could be affected."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["instruction"],
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": (
                            "What to change, as a photographer would say it. "
                            "Name the objects and where they sit, e.g. 'add a "
                            "grey flatweave rug under the bed and a walnut side "
                            "table on the left'."
                        ),
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upscale",
            "description": (
                "Enlarge the current image. Use for requests about resolution, "
                "size or sharpness. Say plainly that resizing cannot add detail "
                "that was never captured."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["scale"],
                "properties": {
                    "scale": {
                        "type": "number",
                        "description": "Multiplier, 1 to 4. Use 2 unless asked otherwise.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repeat_run",
            "description": (
                "Start the whole run again from the same inputs, keeping this "
                "one. Use when the person wants another go at the options "
                "rather than a change to the current image."
            ),
            "parameters": {"type": "object", "additionalProperties": False,
                           "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export",
            "description": "Render the final image into marketplace presets.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["presets"],
                "properties": {
                    "presets": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(PRESETS)},
                    }
                },
            },
        },
    },
]


SYSTEM = """You run an image pipeline for someone preparing their own product listings.

The job runs in one of two modes and they behave differently:

- **edit** - they uploaded a CAD file or photograph. The product is real. Scene
  prompts describe the BACKGROUND ONLY, because the product's own pixels are
  composited back from their file afterwards. Describing the product in a prompt
  wastes a generation.
- **scratch** - no file, only a description. Nothing exists yet, so scene
  prompts describe the whole photograph including the product. There is no
  fidelity check, because there is nothing to be faithful to.

Read `mode` from the job state and write prompts accordingly.

You have the current job state. Use tools to act; answer in plain text when the
person is asking rather than instructing.

How the pipeline works, so your answers are accurate:
- Renders come from their CAD file and carry an exact alpha matte, so the product
  mask needs no segmentation.
- Drafts use a fast model; the final pass uses a slower, more precise one with
  that mask.
- After the edit, the product's pixels are composited back from their source
  file. The model only ever supplies the background. This is why you must never
  describe the product in a scene prompt - describing it invites the model to
  redraw it, and the composite would discard that work anyway.
- QA measures colour difference and structural similarity numerically against the
  raw model output. A fail means the model drifted; the composite still fixes the
  delivered image.

Beyond the two modes there are direct actions on whatever image the run
already has: change_image edits it in place (add props, remove something,
change a surface), upscale enlarges it, repeat_run starts the whole thing over
from the same inputs. Prefer these over a fresh round of drafts when the person
is reacting to an image in front of them.

Style: brief and concrete. One or two sentences unless they asked for detail.
Say what you did, not what you are about to do. Never invent details about their
product that aren't in the brief.

If they ask for something the pipeline can't do - a different product, text or
logos added, a specific model number - say so plainly rather than attempting it.
"""


def _context(state: dict) -> str:
    """Compact snapshot of the job for the model to reason over."""
    brief = state.get("brief") or {}
    drafts = state.get("drafts") or []
    qa = state.get("qa") or {}

    ctx = {
        "stage": state.get("stage"),
        "mode": state.get("mode", "edit"),
        "has_cad_render": bool(state.get("renders")),
        "mask_source": state.get("mask_source"),
        "product": brief.get("product_name"),
        "category": brief.get("category"),
        "must_preserve": brief.get("must_preserve", []),
        "drafts": [{"index": d["index"], "label": d["label"]} for d in drafts],
        "chosen_draft": state.get("chosen_draft"),
        "exports": [e["preset"] for e in state.get("exports", [])],
    }
    if qa:
        ctx["qa"] = {
            "verdict": qa.get("level"),
            "model_output_delta_e": qa.get("stats", {}).get("delta_e_mean"),
            "model_output_ssim": qa.get("stats", {}).get("ssim_product"),
            "after_composite_delta_e": qa.get("residual", {}).get("delta_e_mean"),
            "reviewer_issues": qa.get("review", {}).get("issues", []),
        }
    return json.dumps(ctx)


# --- Tool execution ----------------------------------------------------------

def _run_set_scenes(job_id: str, args: dict, pool) -> str:
    scenes = args["scenes"]
    state = store.get(job_id)
    brief = dict(state.get("brief") or {})
    brief["scene_prompts"] = scenes
    store.update(job_id, brief=brief)
    store.log(job_id, f"Scenes replaced from chat: {', '.join(s['label'] for s in scenes)}.")

    pool.submit(_guarded, job_id, pipeline.drafts_for_mode, job_id)
    noun = "draft" if len(scenes) == 1 else "drafts"
    return f"Generating {len(scenes)} new {noun}: {', '.join(s['label'] for s in scenes)}."


def _run_choose_draft(job_id: str, args: dict, pool) -> str:
    state = store.get(job_id)
    drafts = state.get("drafts") or []
    idx = int(args["index"])
    if not any(d["index"] == idx for d in drafts):
        return f"There's no draft {idx + 1}. Available: 1 to {len(drafts)}."

    label = next(d["label"] for d in drafts if d["index"] == idx)
    instruction = args.get("instruction", "")
    pool.submit(_guarded, job_id, pipeline.finalise, job_id, idx, instruction, None)
    extra = f" with '{instruction}'" if instruction else ""
    return f"Running the final masked edit on {label}{extra}."


def _run_revise_final(job_id: str, args: dict, pool) -> str:
    state = store.get(job_id)
    idx = state.get("chosen_draft")
    if idx is None:
        return "Nothing has been finalised yet, so there's nothing to revise. Pick a draft first."

    instruction = args["instruction"]
    pool.submit(_guarded, job_id, pipeline.finalise, job_id, idx, instruction, None)
    return f"Re-running the final edit: {instruction}"


def _run_export(job_id: str, args: dict, pool) -> str:
    wanted = [p for p in args["presets"] if p in PRESETS]
    if not wanted:
        return "None of those match a preset. Options: " + ", ".join(PRESETS)

    state = store.get(job_id)
    if not state.get("final_file"):
        return "There's no final image yet. Pick a draft and I'll run the edit first."

    pool.submit(_guarded, job_id, pipeline.export, job_id, wanted)
    labels = [PRESETS[p]["label"] for p in wanted]

    warning = ""
    flagged = [PRESETS[p]["label"] for p in wanted if not PRESETS[p]["generative"]]
    if flagged:
        warning = (f" Note that {', '.join(flagged)} expects a real photograph - "
                   "a generated image there risks being rejected.")

    return f"Exporting {', '.join(labels)}.{warning}"


def _run_change_image(job_id: str, args: dict, pool) -> str:
    instruction = args["instruction"]
    if not pipeline.current_image(job_id):
        return "There's no image to work on yet. Start a run first."

    pool.submit(_guarded, job_id, pipeline.enhance, job_id, instruction, "edit", 2.0)
    return (f"Editing the current image: {instruction}\n"
            "Nothing is masked on this path, so the product can change too. "
            "Worth checking against the real thing when it lands.")


def _run_upscale(job_id: str, args: dict, pool) -> str:
    scale = max(1.0, min(4.0, float(args.get("scale", 2))))
    if not pipeline.current_image(job_id):
        return "There's no image to enlarge yet."

    pool.submit(_guarded, job_id, pipeline.enhance, job_id, "", "upscale", scale)
    return (f"Enlarging {scale:g}x. Resizing makes the file bigger but cannot "
            "recover detail that was never in the original.")


def _run_repeat(job_id: str, args: dict, pool) -> str:
    new_id = store.create({"created_at": None,
                           "mode": (store.get(job_id) or {}).get("mode", "edit")})
    pool.submit(_guarded, new_id, pipeline.rerun, job_id, new_id)
    return "Running it again from the same inputs. This one stays in the history."


HANDLERS = {
    "change_image": _run_change_image,
    "upscale": _run_upscale,
    "repeat_run": _run_repeat,
    "set_scenes": _run_set_scenes,
    "choose_draft": _run_choose_draft,
    "revise_final": _run_revise_final,
    "export": _run_export,
}


def _guarded(job_id, fn, *args):
    """Background work from chat fails into the log, never silently."""
    try:
        fn(*args)
    except Exception as exc:
        store.log(job_id, f"{type(exc).__name__}: {exc}", "error")
        store.update(job_id, stage="failed", error=str(exc))


# --- Entry point -------------------------------------------------------------

def respond(job_id: str, message: str, pool) -> dict:
    state = store.get(job_id)
    if not state:
        raise KeyError(job_id)

    history = state.get("chat", [])
    history.append({"role": "user", "content": message})

    if models.live():
        reply, actions = _live_turn(job_id, message, history, state, pool)
    else:
        reply, actions = _offline_turn(job_id, message, state, pool)

    history.append({"role": "assistant", "content": reply})
    store.update(job_id, chat=history[-(MAX_HISTORY * 2):])
    return {"reply": reply, "actions": actions}


def _live_turn(job_id, message, history, state, pool):
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "system", "content": "Current job state: " + _context(state)},
    ]
    messages += [m for m in history[-(MAX_HISTORY * 2):]]

    kwargs = {
        "model": config.REASONING_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "max_completion_tokens": 2000,
    }

    # gpt-6-astra refuses function tools on /v1/chat/completions unless
    # reasoning_effort is explicitly "none" - and omitting it doesn't help,
    # because the model then applies its default effort and the same refusal
    # comes back. So "none" first, then a bare call, then the configured
    # effort for any model that wants it the other way round.
    attempts = [
        {**kwargs, "reasoning_effort": "none"},
        kwargs,
        {**kwargs, "reasoning_effort": config.REASONING_EFFORT},
    ]

    resp = None
    last = None
    for attempt in attempts:
        try:
            resp = client.chat.completions.create(**attempt)
            break
        except Exception as exc:
            last = exc
    if resp is None:
        raise last

    choice = resp.choices[0].message
    actions = []
    notes = []

    for call in (choice.tool_calls or []):
        name = call.function.name
        handler = HANDLERS.get(name)
        if not handler:
            continue
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            continue
        notes.append(handler(job_id, args, pool))
        actions.append(name)

    reply = (choice.content or "").strip()
    if notes:
        reply = (reply + "\n\n" if reply else "") + "\n".join(notes)
    return reply or "Done.", actions


# --- Offline fallback --------------------------------------------------------
#
# Mock mode has no model to route through, so this reads the message directly.
# Deliberately simple: it covers the handful of instructions worth testing the
# UI with, and says so when it doesn't understand rather than guessing.

ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3,
            "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "one": 0, "two": 1, "three": 2}


def _offline_turn(job_id, message, state, pool):
    text = message.lower().strip()

    # Export?
    named = [k for k in PRESETS if k.replace("_", " ") in text or k in text]
    for word, key in [("amazon", "amazon_secondary"), ("etsy", "etsy"),
                      ("shopify", "shopify"), ("ebay", "ebay"),
                      ("transparent", "transparent_png"), ("4k", "hero_4k"),
                      ("hero", "hero_4k")]:
        if word in text and key not in named:
            named.append(key)

    if named and any(w in text for w in ("export", "download", "give me", "generate", "make", "need")):
        return _run_export(job_id, {"presets": named}, pool), ["export"]

    # Pick a draft?
    idx = None
    m = re.search(r"\b(?:draft|option|number|no\.?)\s*(\d)", text)
    if m:
        idx = int(m.group(1)) - 1
    else:
        for word, i in ORDINALS.items():
            if re.search(rf"\b{word}\b", text) and any(
                    w in text for w in ("draft", "option", "go with", "use", "pick", "choose", "take")):
                idx = i
                break
        if idx is None:
            for d in state.get("drafts", []):
                if d["label"].lower() in text:
                    idx = d["index"]
                    break

    if idx is not None:
        return _run_choose_draft(job_id, {"index": idx, "instruction": ""}, pool), ["choose_draft"]

    # Enlarge?
    if any(w in text for w in ("upscale", "bigger", "resolution", "larger", "enlarge", "sharper")):
        import re as _re
        m2 = _re.search(r"(\d(?:\.\d)?)\s*x", text)
        return _run_upscale(job_id, {"scale": float(m2.group(1)) if m2 else 2}, pool), ["upscale"]

    # Run it again?
    if any(w in text for w in ("run again", "rerun", "re-run", "try again", "another go")):
        return _run_repeat(job_id, {}, pool), ["repeat_run"]

    # Add or remove something in the picture?
    if state.get("final_file") and any(
            w in text for w in ("add ", "remove ", "put ", "take out", "prop", "rug",
                                "table", "lamp", "plant", "cushion", "chair")):
        return _run_change_image(job_id, {"instruction": message}, pool), ["change_image"]

    # Revise the finished image?
    if state.get("final_file") and any(
            w in text for w in ("shadow", "light", "brighter", "darker", "warmer",
                                "cooler", "softer", "harder", "change", "adjust", "less", "more")):
        return _run_revise_final(job_id, {"instruction": message}, pool), ["revise_final"]

    # Describe a scene?
    scene_words = ("background", "backdrop", "wood", "marble", "concrete", "studio",
                   "white", "table", "counter", "surface", "outdoor", "kitchen",
                   "desk", "shelf", "linen", "stone", "scene", "setting")
    if any(w in text for w in scene_words):
        return _run_set_scenes(job_id, {"scenes": [{"label": "From chat", "prompt": message}]}, pool), ["set_scenes"]

    # Otherwise: report, don't guess.
    stage = state.get("stage")
    qa = state.get("qa") or {}
    if "qa" in text or "quality" in text or "why" in text:
        if qa:
            s = qa.get("stats", {})
            return (f"Verdict {qa.get('level')}. The model's raw output drifted "
                    f"{s.get('delta_e_mean')} dE2000 with SSIM {s.get('ssim_product')}. "
                    f"After compositing the residual is "
                    f"{qa.get('residual', {}).get('delta_e_mean')} dE, so the delivered "
                    "image is clean."), []
        return "No quality check has run yet.", []

    return (f"Mock mode has no model behind the chat, so it only understands direct "
            f"instructions: describe a background, say which draft to use, ask to "
            f"revise the final, or name export presets. Current stage: {stage}."), []
