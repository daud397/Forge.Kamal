"""The pipeline itself, one function per stage.

Stages run in a worker thread and write progress to the job store, so the
browser can poll and the tab can be closed without losing the run.
"""
from __future__ import annotations

import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import cad, config, imaging, models, store
from presets import preset


# Long edge cap for uploaded photographs.
PHOTO_MAX_EDGE = 2048


def job_dir(job_id: str) -> Path:
    d = config.DATA / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(job_id: str, img: Image.Image, name: str) -> str:
    path = job_dir(job_id) / name
    img.save(path)
    return name


def _load(job_id: str, name: str) -> Image.Image:
    return Image.open(job_dir(job_id) / name).convert("RGBA")


def _thumb(job_id: str):
    """Small still for the run history.

    Picks the most finished thing the run has produced, so a history entry
    shows the delivered image where there is one and the starting point where
    there isn't - rather than a row of identical placeholder icons.
    """
    state = store.get(job_id) or {}
    drafts = state.get("drafts") or []
    source = (state.get("final_file")
              or (drafts[0]["file"] if drafts else None)
              or state.get("base_image"))
    if not source:
        return
    try:
        img = imaging.flatten(_load(job_id, source), (126, 126, 126))
        img.thumbnail((320, 320), Image.LANCZOS)
        img.save(job_dir(job_id) / "thumb.jpg", "JPEG", quality=82)
        store.update(job_id, thumb="thumb.jpg")
    except Exception:
        pass            # a missing thumbnail is never worth failing a run over


# --- Stage 1: ingest ---------------------------------------------------------

def ingest(job_id: str, uploads: list[tuple[str, bytes]], note: str = "") -> dict:
    """Sort uploads into CAD and photographs, render the CAD, pick a base frame."""
    d = job_dir(job_id)
    store.update(job_id, stage="ingesting")

    cad_files, photos = [], []
    for filename, blob in uploads:
        suffix = Path(filename).suffix.lower()
        dest = d / f"src_{Path(filename).name}"
        dest.write_bytes(blob)
        if suffix in config.CAD_EXTS:
            cad_files.append(dest)
        elif suffix in config.IMAGE_EXTS:
            photos.append(dest)
        else:
            store.log(job_id, f"Skipped {filename}: unsupported extension.", "warn")

    renders: dict[str, str] = {}
    for path in cad_files:
        store.log(job_id, f"Rendering {path.name}. Meshes over {config.MAX_FACES} faces are decimated first.")
        t0 = time.time()
        try:
            angles = cad.render_angles(path, d)
            renders.update(angles)
            store.log(job_id, f"Rendered {len(angles)} angles in {time.time() - t0:.1f}s.")
        except cad.CADError as exc:
            store.log(job_id, str(exc), "error")

    photo_names = []
    for path in photos:
        img = Image.open(path).convert("RGBA")

        # A phone photo can be 4000px on the long edge. The final pass renders
        # at 1536 and the composite happens there, so anything beyond about
        # 2048 is weight we pay to upload and store for no gain in the output.
        if max(img.size) > PHOTO_MAX_EDGE:
            k = PHOTO_MAX_EDGE / max(img.size)
            before = img.size
            img = imaging.resize_rgba(
                img, (round(img.width * k), round(img.height * k)))
            store.log(job_id, f"{path.name} resized from {before[0]}x{before[1]} "
                              f"to {img.width}x{img.height}.")

        name = f"photo_{path.stem}.png"
        img.save(d / name)
        photo_names.append(name)

    # The hero render is the best base: it carries an exact alpha matte.
    if "hero" in renders:
        base = renders["hero"]
        mask_source = "cad_alpha"
    elif renders:
        base = next(iter(renders.values()))
        mask_source = "cad_alpha"
    elif photo_names:
        base = photo_names[0]
        mask_source = "needs_paint"
    else:
        raise ValueError("Nothing usable was uploaded. Add a CAD file or a product photo.")

    return store.update(
        job_id,
        stage="ingested",
        note=note,
        renders=renders,
        photos=photo_names,
        base_image=base,
        mask_source=mask_source,
        created_at=time.strftime("%Y-%m-%d %H:%M"),
    )


# --- Stage 1b: start from a description, no upload --------------------------

def describe(job_id: str, description: str) -> dict:
    """No file, no render. Astra writes the brief from the seller's sentence."""
    store.update(job_id, stage="briefing")
    store.log(job_id, f"No source file. {config.REASONING_MODEL} writing a brief "
                      "from the description.")

    brief = models.invent(description)
    store.log(job_id, f"Brief: {brief.get('product_name')} - "
                      f"{len(brief.get('scene_prompts', []))} treatments.")

    return store.update(
        job_id,
        stage="briefed",
        mode="scratch",
        note=description,
        brief=brief,
        renders={},
        photos=[],
        base_image=None,
        mask_source="none",
        created_at=time.strftime("%Y-%m-%d %H:%M"),
    )


def drafts_scratch(job_id: str) -> dict:
    """Text to image. Nothing is being preserved, so there is no mask."""
    state = store.get(job_id)
    brief = state.get("brief") or {}
    store.update(job_id, stage="drafting")

    scenes = brief.get("scene_prompts", [])[:config.DRAFT_COUNT]
    if not scenes:
        raise ValueError("The brief contains no scenes to generate.")

    note = state.get("note", "")
    size = gen_size(note, config.DRAFT_SIZE)

    store.log(job_id, f"{config.DRAFT_MODEL} generating {len(scenes)} images at "
                      f"{size[0]}x{size[1]}.")

    results = models.generate_scratch([sc["prompt"] for sc in scenes], size,
                                      directive=note, brief=brief)
    out = []
    for i, (prompt, img) in enumerate(results):
        name = _save(job_id, img, f"draft_{i}.png")
        out.append({"index": i, "label": scenes[i].get("label", f"Option {i+1}"),
                    "prompt": prompt, "file": name})

    store.log(job_id, f"{len(out)} images ready.")
    result = store.update(job_id, stage="drafted", drafts=out)
    _thumb(job_id)
    return result


# --- Stage 2: analyse --------------------------------------------------------

def analyse(job_id: str) -> dict:
    state = store.get(job_id)
    store.update(job_id, stage="analysing")

    frames = [_load(job_id, n) for n in
              list(state.get("renders", {}).values())[:3] + state.get("photos", [])[:3]]

    store.log(job_id, f"{config.REASONING_MODEL} reading {len(frames)} frame(s).")
    brief = models.analyse(frames, state.get("note", ""))
    store.log(job_id, f"Brief: {brief.get('product_name')} - {len(brief.get('scene_prompts', []))} scene options.")

    return store.update(job_id, stage="briefed", brief=brief)


# --- Stage 3: drafts ---------------------------------------------------------

def drafts_for_mode(job_id: str) -> dict:
    """Regenerate using whichever path this job started on."""
    state = store.get(job_id)
    return drafts_scratch(job_id) if state.get("mode") == "scratch" else drafts(job_id)


def gen_size(note: str, default: tuple[int, int]) -> tuple[int, int]:
    """Honour a requested aspect ratio, falling back to the configured size."""
    ratio = imaging.ratio_from_text(note)
    if ratio is None:
        return default
    return imaging.size_for_ratio(ratio, default[0] * default[1])


def drafts(job_id: str) -> dict:
    state = store.get(job_id)
    brief = state.get("brief") or {}
    store.update(job_id, stage="drafting")

    sources = [_load(job_id, n) for n in (state.get("photos") or [])] \
        or [_load(job_id, state["base_image"])]
    prompts = [p["prompt"] for p in brief.get("scene_prompts", [])][:config.DRAFT_COUNT]
    labels = [p.get("label", f"Option {i+1}") for i, p in enumerate(brief.get("scene_prompts", []))]

    if not prompts:
        raise ValueError("The brief contains no scene prompts to draft from.")

    note = state.get("note", "")
    size = gen_size(note, config.DRAFT_SIZE)
    if size != config.DRAFT_SIZE:
        store.log(job_id, f"You asked for a specific aspect ratio, so these are "
                          f"{size[0]}x{size[1]}.")

    store.log(job_id, f"{config.DRAFT_MODEL} generating {len(prompts)} drafts at "
                      f"{size[0]}x{size[1]}, quality {config.DRAFT_QUALITY}.")

    if len(sources) > 1:
        store.log(job_id, f"Sending all {len(sources)} images to the model in the "
                          "order you attached them - 'first image' means the first "
                          "one you attached.")
    results = models.generate_drafts(sources, prompts, size, directive=note,
                                     brief=brief)
    out = []
    for i, (prompt, img) in enumerate(results):
        name = _save(job_id, img, f"draft_{i}.png")
        out.append({"index": i, "label": labels[i] if i < len(labels) else f"Option {i+1}",
                    "prompt": prompt, "file": name})

    store.log(job_id, f"{len(out)} drafts ready for selection.")
    result = store.update(job_id, stage="drafted", drafts=out)
    _thumb(job_id)
    return result


# --- Stage 4: final masked edit + QA ----------------------------------------

def finalise(job_id: str, draft_index: int, extra_instruction: str = "",
             painted_mask: bytes | None = None) -> dict:
    state = store.get(job_id)
    store.update(job_id, stage="finalising")

    chosen = next((d for d in state.get("drafts", []) if d["index"] == draft_index), None)
    if not chosen:
        raise ValueError(f"No draft with index {draft_index}.")

    scratch = state.get("mode") == "scratch"
    multi_ref = len(state.get("photos") or []) > 1

    # In scratch mode the approved draft IS the source: the refinement pass
    # works on the image the seller picked, not on any uploaded file.
    base = _load(job_id, chosen["file"] if scratch else state["base_image"])
    size = gen_size(state.get("note", ""), config.FINAL_SIZE)

    photos = state.get("photos") or []
    extra_refs = ([_load(job_id, n) for n in photos[1:4]]
                  if not scratch and len(photos) > 1 else None)

    # Where does the product mask come from?
    if scratch:
        product_mask = None
        store.log(job_id, "Generated image, so there is nothing to protect. "
                          "The refinement pass runs unmasked.")
    elif painted_mask:
        product_mask = imaging.mask_from_strokes(painted_mask, base.size)
        store.log(job_id, "Using the painted mask as the preserved region.")
    elif state.get("mask_source") == "cad_alpha":
        product_mask = imaging.mask_from_alpha(base)
        store.log(job_id, "Preserved region taken from the CAD alpha matte - pixel exact, no segmentation.")
    else:
        product_mask = None
        store.log(job_id, "No mask available. The edit runs unmasked and the product is not protected.", "warn")

    mask_png = imaging.edit_mask(product_mask, size) if product_mask else None
    if scratch:
        prompt_lead = "Refine this photograph. Keep the product, its colour, "\
                      "material and framing exactly as they are."
    else:
        prompt_lead = None

    prompt = prompt_lead if prompt_lead else chosen["prompt"]
    if extra_instruction.strip():
        prompt = f"{prompt}\n\nAdditional direction: {extra_instruction.strip()}"

    store.log(job_id, f"{config.FINAL_MODEL} editing at {size[0]}x{size[1]}, quality {config.FINAL_QUALITY}.")
    edited = models.final_edit(base, prompt, mask_png, size, extra_refs=extra_refs)
    _save(job_id, edited, "final_raw.png")

    # Hard guarantee, not a hope: the product pixels come from the source.
    if product_mask:
        composited = imaging.composite_preserve(base, edited, product_mask)
        store.log(job_id, "Composited the source product back over the generated background.")
    else:
        composited = edited

    _save(job_id, composited, "final_composited.png")

    # Measured QA.
    #
    # Grade `edited`, not `composited`. The composite pastes the source product
    # back, so grading it would grade the fix rather than the model - a run
    # where Sunburst mangled the product would still score a clean pass. The
    # raw comparison is what tells you whether the masked edit actually held.
    measure_mask = imaging.measurement_mask(product_mask) if product_mask \
        else Image.new("L", composited.size, 255)

    stats = imaging.region_stats(base, edited, measure_mask)

    if scratch:
        # Nothing was supplied to be faithful to, so colour difference against
        # the draft measures how much the refinement changed - which is often
        # the point. Reporting it as a failure would be meaningless. The numbers
        # stay visible as information; the verdict comes from the reviewer.
        level = "info"
        notes = [f"Refinement moved the image {stats['delta_e_mean']} dE2000 on "
                 f"average from the draft you approved. There is no source file "
                 f"to check fidelity against, so this is information, not a test."]
    elif multi_ref:
        # The output is a composition of several references, so comparing it
        # against the first upload alone would flag exactly the changes that
        # were asked for. The numbers stay; the verdict doesn't apply.
        level = "info"
        notes = ["Composed from several reference images, so there is no single "
                 "source to measure against. Check the design against your "
                 "reference by eye."]
    else:
        level, notes = imaging.verdict(stats)
    store.log(job_id, f"Model output: dE2000 mean {stats['delta_e_mean']}, "
                      f"p95 {stats['delta_e_p95']}, SSIM {stats['ssim_product']} -> {level}.")

    # What survives after compositing. Should be near zero; if it isn't, the
    # mask and the render have drifted out of alignment.
    residual = imaging.region_stats(base, composited, measure_mask)
    if not scratch:
        store.log(job_id, f"After composite: dE2000 mean {residual['delta_e_mean']}, "
                          f"SSIM {residual['ssim_product']}.")

    if not scratch and level != "pass" and residual["delta_e_mean"] < config.DELTA_E_PASS:
        notes.append(
            "The composite corrected this: delivered product pixels come from your "
            "source file, so the shipped image is clean. The drift is a signal that "
            "the prompt or mask needs work, not a reason to reject this export."
        )

    # Judged QA.
    if config.QA_REVIEW:
        review = models.qa_review(composited, state.get("brief") or {})
    else:
        review = {"usable": True, "issues": [], "fix_instruction": ""}
    if scratch and not review.get("usable", True):
        level = "warn"
    if review.get("issues"):
        for issue in review["issues"]:
            store.log(job_id, f"Review: {issue}", "warn")

    updated = store.update(
        job_id,
        stage="finalised",
        chosen_draft=draft_index,
        final_file="final_composited.png",
        qa={"stats": stats, "residual": residual, "level": level,
            "notes": notes, "review": review},
    )
    _thumb(job_id)
    return updated


# --- Taking a draft as it stands ---------------------------------------------

def use_as_is(job_id: str, draft_index: int) -> dict:
    """Promote a draft to final without generating anything.

    The final pass exists to add fidelity the drafts lack, but a draft is often
    already the shot - the composition is right and nothing needs changing.
    Running a generation to arrive back where you started costs money and can
    only move the image away from what you approved.

    No model call happens here at all. Exports still enlarge with Lanczos, so
    the delivered files are full size.
    """
    state = store.get(job_id)
    chosen = next((d for d in (state.get("drafts") or [])
                   if d["index"] == draft_index), None)
    if not chosen:
        raise ValueError(f"There is no option {draft_index + 1} in this run.")

    store.update(job_id, stage="finalising")
    store.log(job_id, f"Taking '{chosen['label']}' as it stands. No generation, "
                      "so nothing costs anything and nothing can drift.")

    img = _load(job_id, chosen["file"])
    _save(job_id, img, "final_composited.png")

    updated = store.update(
        job_id,
        stage="finalised",
        chosen_draft=draft_index,
        final_file="final_composited.png",
        used_as_is=True,
        qa={"stats": {"delta_e_mean": 0.0, "delta_e_p95": 0.0, "delta_e_max": 0.0,
                      "ssim_product": 1.0, "product_coverage": 1.0,
                      "pixels_out_of_tolerance": 0},
            "level": "info",
            "notes": ["Taken straight from the option you picked. Nothing was "
                      "regenerated, so this is exactly the image you chose."],
            "review": {"usable": True, "issues": [], "fix_instruction": ""}},
    )
    _thumb(job_id)
    return updated


def redraft(job_id: str) -> dict:
    """Another set of options from the same brief.

    The models are stochastic, so the same prompts give different photographs.
    This is the cheap way to say "not these" without rewriting the brief.
    """
    state = store.get(job_id)
    if not (state.get("brief") or {}).get("scene_prompts"):
        raise ValueError("This run has no brief to generate from.")

    store.log(job_id, "Generating another set from the same brief.")
    store.update(job_id, final_file=None, chosen_draft=None, qa=None)
    return drafts_for_mode(job_id)


# --- Working on an image directly -------------------------------------------

def current_image(job_id: str) -> str | None:
    """The most worked-on image a run has: final, else chosen draft, else source."""
    state = store.get(job_id) or {}
    if state.get("final_file"):
        return state["final_file"]
    drafts = state.get("drafts") or []
    chosen = state.get("chosen_draft")
    if drafts:
        pick = next((d for d in drafts if d["index"] == chosen), drafts[0])
        return pick["file"]
    return state.get("base_image")


def enhance(job_id: str, instruction: str, action: str = "edit",
            scale: float = 2.0, region_png: bytes | None = None,
            detail: bool | None = None, source_file: str | None = None) -> dict:
    """Change or enlarge the image that already exists, with no drafting round.

    This is the path for "add a grey rug", "take the lamp out", "make it
    bigger" - a direct instruction against the current image rather than a new
    run. Each call stacks on the last result, so edits accumulate the way they
    would in a photo editor.
    """
    state = store.get(job_id)

    # An explicit file wins - that is how editing the second draft edits the
    # second draft rather than whatever the run considers current. It has to be
    # a file this run owns; anything else is a path we refuse to touch.
    if source_file:
        if "/" in source_file or "\\" in source_file or ".." in source_file \
                or not (job_dir(job_id) / source_file).exists():
            raise ValueError("That image doesn't belong to this run.")
        source_name = source_file
    else:
        source_name = current_image(job_id)
    if not source_name:
        raise ValueError("This run has no image to work on yet.")

    store.update(job_id, stage="finalising")
    source = _load(job_id, source_name)

    versions = list(state.get("versions") or [])
    step = len(versions) + 1

    if action == "upscale":
        result = _upscale(job_id, source, scale, detail)
    elif region_png:
        result = _edit_region(job_id, source, instruction, region_png)
    else:
        result = _edit_freely(job_id, source, instruction)

    name = _save(job_id, result, f"v{step}.png")
    versions.append({
        "step": step, "file": name, "action": action,
        "instruction": instruction or ("%.1fx" % scale),
        "size": f"{result.width}x{result.height}",
    })

    _save(job_id, result, "final_composited.png")
    updated = store.update(
        job_id,
        stage="finalised",
        final_file="final_composited.png",
        versions=versions,
        qa={"stats": {"delta_e_mean": 0, "delta_e_p95": 0, "delta_e_max": 0,
                      "ssim_product": 1.0, "product_coverage": 1.0,
                      "pixels_out_of_tolerance": 0},
            "level": "info",
            "notes": ["Worked on directly, so there is no source to measure "
                      "against. Judge this one by eye."],
            "review": {"usable": True, "issues": [], "fix_instruction": ""}},
    )
    _thumb(job_id)
    return updated


def _edit_freely(job_id: str, source: Image.Image, instruction: str) -> Image.Image:
    """Full-frame edit. Nothing is masked, so the model may touch anything."""
    if not instruction.strip():
        raise ValueError("Say what to change, for example: add a grey rug and a "
                         "modern side table.")

    store.log(job_id, f"{config.FINAL_MODEL} editing the whole frame: "
                      f"{instruction.strip()[:80]}")
    store.log(job_id, "Nothing is masked on this path, so the product itself can "
                      "change. Check the result against the real thing.", "warn")

    prompt = (
        f"{instruction.strip()}\n\n"
        "Keep the existing subject, its colour, material, proportions and "
        "position exactly as they are. Match the lighting direction, colour "
        "temperature and shadow behaviour already in the photograph so anything "
        "added looks like it was there when the shot was taken. Photographic, "
        "no text, no logos, no watermarks."
    )
    return models.final_edit(source, prompt, None, config.FINAL_SIZE)


def _edit_region(job_id: str, source: Image.Image, instruction: str,
                 region_png: bytes) -> Image.Image:
    """Change only what was painted, and put the rest back pixel for pixel.

    Two things happen here. The mask tells the model which area it may touch,
    and the composite afterwards guarantees the rest is untouched - because a
    masked edit still re-encodes the whole frame, so without the composite the
    untouched areas drift slightly on every pass. Over three or four edits that
    drift is visible.
    """
    if not instruction.strip():
        raise ValueError("Say what should change in the area you painted.")

    size = config.FINAL_SIZE
    editable = imaging.mask_from_strokes(region_png, source.size)

    # mask_from_strokes marks what was painted. Here painted means "change
    # this", which is the opposite of the product mask, so it gets inverted
    # before being handed over as the region to keep.
    keep = Image.fromarray(255 - np.asarray(editable))

    store.log(job_id, f"{config.FINAL_MODEL} editing the painted area only: "
                      f"{instruction.strip()[:80]}")

    mask_png = imaging.edit_mask(keep, size)
    edited = models.final_edit(
        source,
        f"{instruction.strip()}\n\nChange only the masked area. Match the "
        "lighting direction, colour temperature, grain and shadow behaviour of "
        "the surrounding photograph so the edit is invisible at the seam. No "
        "text, no logos, no watermarks.",
        mask_png, size)

    result = imaging.composite_preserve(source, edited, keep, feather=2.0)
    store.log(job_id, "Everything outside the painted area restored from the "
                      "previous version.")
    return result


def _upscale(job_id: str, source: Image.Image, scale: float,
             detail: bool | None = None) -> Image.Image:
    """Enlarge, optionally asking the model to restore detail on the way.

    Plain Lanczos makes an image bigger without making it better - it cannot
    add detail that was never captured. The detail pass genuinely can, at the
    cost of being a generative step: it may quietly change what it is
    sharpening. Both are offered because which one you want depends on whether
    the image is a real photograph of a real product.
    """
    w, h = source.size
    target = (min(config.MAX_EDGE, int(w * scale)),
              min(config.MAX_EDGE, int(h * scale)))

    if config.UPSCALE_DETAIL_PASS if detail is None else detail:
        gen_w, gen_h = imaging.snap_size(*target)
        store.log(job_id, f"{config.FINAL_MODEL} restoring detail at {gen_w}x{gen_h}.")
        try:
            source = models.final_edit(
                source,
                "Restore and sharpen fine detail at higher resolution. Do not "
                "change the composition, the colour, the materials, or any "
                "object in the frame. Add nothing and remove nothing.",
                None, (gen_w, gen_h))
        except Exception as exc:
            store.log(job_id, f"Detail pass failed ({exc}). Falling back to a "
                              "plain resize.", "warn")

    store.log(job_id, f"Resizing to {target[0]}x{target[1]} with Lanczos, "
                      "which invents nothing.")
    return imaging.upscale(source, target, fit="contain").convert("RGB")


# --- Re-running --------------------------------------------------------------

def rerun(job_id: str, new_job_id: str) -> dict:
    """Start a fresh run from an old one's inputs, keeping the original intact.

    Copying rather than overwriting matters: the point of re-running is usually
    that you want a different result, and you only know it is better if the
    first one is still there to compare against.
    """
    old = store.get(job_id)
    if not old:
        raise ValueError("No such run to repeat.")

    src_dir = job_dir(job_id)
    job_dir(new_job_id)                       # create the destination up front
    uploads = []
    for path in sorted(src_dir.glob("src_*")):
        uploads.append((path.name.replace("src_", "", 1), path.read_bytes()))

    store.log(new_job_id, f"Repeating run {job_id}.")

    if uploads:
        ingest(new_job_id, uploads, old.get("note", ""))
        analyse(new_job_id)
        return drafts(new_job_id)

    # A described run has no files, so repeat the description instead.
    description = old.get("note") or (old.get("brief") or {}).get("product_name")
    if not description:
        raise ValueError("That run has nothing to repeat from.")
    describe(new_job_id, description)
    return drafts_scratch(new_job_id)


# --- Stage 5: export ---------------------------------------------------------

def export(job_id: str, preset_names: list[str]) -> dict:
    state = store.get(job_id)
    store.update(job_id, stage="exporting")

    src = _load(job_id, state.get("final_file") or state["base_image"])
    d = job_dir(job_id)
    written = []

    for name in preset_names:
        spec = preset(name)
        target = spec["size"]
        bg = "transparent" if spec["background"] == "transparent" else "white"

        out = imaging.upscale(src, target, fit="contain", background=bg)

        fmt = spec["format"].upper()
        ext = {"JPEG": "jpg", "WEBP": "webp", "PNG": "png"}[fmt]
        filename = f"export_{name}.{ext}"

        if fmt == "JPEG":
            out.convert("RGB").save(d / filename, "JPEG", quality=92, subsampling=1)
        elif fmt == "WEBP":
            out.save(d / filename, "WEBP", quality=92, method=5)
        else:
            out.save(d / filename, "PNG")

        scale = target[0] / src.width
        written.append({
            "preset": name,
            "label": spec["label"],
            "file": filename,
            "size": f"{target[0]}x{target[1]}",
            "scale": f"{scale:.2f}x Lanczos",
            "generative_allowed": spec["generative"],
            "note": spec["note"],
        })
        store.log(job_id, f"{spec['label']}: {target[0]}x{target[1]} via {scale:.2f}x Lanczos, no generative pass.")

    zip_path = d / "exports.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for item in written:
            z.write(d / item["file"], item["file"])

    return store.update(job_id, stage="exported", exports=written, zip_file="exports.zip")


# --- Intent: what did they actually ask for? ---------------------------------
#
# A file plus a sentence is not always "put this in a scene". If the sentence
# says upscale, or says add a rug, running the scene flow ignores them and
# spends money on drafts nobody asked for. Keyword matching here is deliberate:
# it is instant and free, and the phrasings are not subtle.

UPSCALE_WORDS = ("resolution", "upscale", "blurry", "blur", "sharpen", "sharper",
                 "pixel", "bigger", "larger", "enlarge", "hi-res", "hires",
                 "high res", "quality", "crisp", "clearer")
EDIT_WORDS = ("add ", "remove ", "take out", "replace ", "change ", "put ",
              "swap ", "make the ", "make it ", "turn the ", "recolour", "recolor",
              "prop", "rug", "lamp", "table", "plant", "cushion", "pillow", "curtain")
SCENE_WORDS = ("scene", "background", "backdrop", "roomset", "room set",
               "lifestyle", "setting", "studio", "options", "drafts", "variations")


def intent_of(note: str) -> str:
    """'upscale', 'edit' or 'scene'."""
    t = (note or "").lower()
    if not t.strip():
        return "scene"
    if any(w in t for w in SCENE_WORDS):
        return "scene"
    if any(w in t for w in UPSCALE_WORDS):
        return "upscale"
    if any(w in t for w in EDIT_WORDS):
        return "edit"
    return "scene"


# --- Runner ------------------------------------------------------------------

def run_auto(job_id: str, uploads: list[tuple[str, bytes]], note: str):
    """Ingest, then do what the note asked for.

    A direct instruction skips the brief and the drafts entirely. That is two
    fewer model calls and no waiting on options that were never wanted.
    """
    try:
        ingest(job_id, uploads, note)
        kind = intent_of(note)

        if kind == "upscale":
            store.log(job_id, "You asked for resolution, so this goes straight to "
                              "a detail pass and enlargement - no scenes.")
            store.update(job_id, intent="upscale")
            enhance(job_id, "", "upscale", 2.0, detail=True)
            return

        if kind == "edit":
            store.log(job_id, "You asked for a change, so this edits the photo "
                              "directly - no scenes.")
            store.update(job_id, intent="edit")
            enhance(job_id, note, "edit", 2.0)
            return

        store.update(job_id, intent="scene")
        analyse(job_id)
        drafts(job_id)
    except Exception as exc:
        store.log(job_id, f"{type(exc).__name__}: {exc}", "error")
        store.log(job_id, traceback.format_exc(limit=3), "error")
        store.update(job_id, stage="failed", error=str(exc))


def run_describe(job_id: str, description: str):
    """Description through to drafts, then stop for a human to choose."""
    try:
        describe(job_id, description)
        drafts_scratch(job_id)
    except Exception as exc:
        store.log(job_id, f"{type(exc).__name__}: {exc}", "error")
        store.update(job_id, stage="failed", error=str(exc))


def run_enhance(job_id: str, instruction: str, action: str, scale: float,
                presets_wanted: list[str] | None = None,
                region_png: bytes | None = None, detail: bool | None = None,
                source_file: str | None = None):
    try:
        enhance(job_id, instruction, action, scale, region_png, detail, source_file)
        if presets_wanted:
            export(job_id, presets_wanted)
    except Exception as exc:
        store.log(job_id, f"{type(exc).__name__}: {exc}", "error")
        store.update(job_id, stage="failed", error=str(exc))


def run_use_as_is(job_id: str, draft_index: int,
                  presets_wanted: list[str] | None = None):
    try:
        use_as_is(job_id, draft_index)
        if presets_wanted:
            export(job_id, presets_wanted)
    except Exception as exc:
        store.log(job_id, f"{type(exc).__name__}: {exc}", "error")
        store.update(job_id, stage="failed", error=str(exc))


def run_redraft(job_id: str):
    try:
        redraft(job_id)
    except Exception as exc:
        store.log(job_id, f"{type(exc).__name__}: {exc}", "error")
        store.update(job_id, stage="failed", error=str(exc))


def run_rerun(job_id: str, new_job_id: str):
    try:
        rerun(job_id, new_job_id)
    except Exception as exc:
        store.log(new_job_id, f"{type(exc).__name__}: {exc}", "error")
        store.update(new_job_id, stage="failed", error=str(exc))


def run_finalise(job_id: str, draft_index: int, instruction: str,
                 painted_mask: bytes | None, presets_wanted: list[str]):
    try:
        finalise(job_id, draft_index, instruction, painted_mask)
        if presets_wanted:
            export(job_id, presets_wanted)
    except Exception as exc:
        store.log(job_id, f"{type(exc).__name__}: {exc}", "error")
        store.log(job_id, traceback.format_exc(limit=3), "error")
        store.update(job_id, stage="failed", error=str(exc))
