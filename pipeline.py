"""The pipeline itself, one function per stage.

Stages run in a worker thread and write progress to the job store, so the
browser can poll and the tab can be closed without losing the run.
"""
from __future__ import annotations

import io
import time
import traceback
import zipfile
from pathlib import Path

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

    store.log(job_id, f"{config.DRAFT_MODEL} generating {len(scenes)} images at "
                      f"{config.DRAFT_SIZE[0]}x{config.DRAFT_SIZE[1]}.")

    results = models.generate_scratch([sc["prompt"] for sc in scenes])
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


def drafts(job_id: str) -> dict:
    state = store.get(job_id)
    brief = state.get("brief") or {}
    store.update(job_id, stage="drafting")

    base = _load(job_id, state["base_image"])
    prompts = [p["prompt"] for p in brief.get("scene_prompts", [])][:config.DRAFT_COUNT]
    labels = [p.get("label", f"Option {i+1}") for i, p in enumerate(brief.get("scene_prompts", []))]

    if not prompts:
        raise ValueError("The brief contains no scene prompts to draft from.")

    store.log(job_id, f"{config.DRAFT_MODEL} generating {len(prompts)} drafts at "
                      f"{config.DRAFT_SIZE[0]}x{config.DRAFT_SIZE[1]}, quality {config.DRAFT_QUALITY}.")

    results = models.generate_drafts(base, prompts)
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

    # In scratch mode the approved draft IS the source: the refinement pass
    # works on the image the seller picked, not on any uploaded file.
    base = _load(job_id, chosen["file"] if scratch else state["base_image"])
    size = config.FINAL_SIZE

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
    edited = models.final_edit(base, prompt, mask_png, size)
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
    review = models.qa_review(composited, state.get("brief") or {})
    if scratch and not review.get("usable", True):
        level = "warn"
    if review.get("issues"):
        for issue in review["issues"]:
            store.log(job_id, f"Review: {issue}", "warn")

    return store.update(
        job_id,
        stage="finalised",
        chosen_draft=draft_index,
        final_file="final_composited.png",
        qa={"stats": stats, "residual": residual, "level": level,
            "notes": notes, "review": review},
    )
    _thumb(job_id)
    return store.get(job_id)


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
            scale: float = 2.0) -> dict:
    """Change or enlarge the image that already exists, with no drafting round.

    This is the path for "add a grey rug", "take the lamp out", "make it
    bigger" - a direct instruction against the current image rather than a new
    run. Each call stacks on the last result, so edits accumulate the way they
    would in a photo editor.
    """
    state = store.get(job_id)
    source_name = current_image(job_id)
    if not source_name:
        raise ValueError("This run has no image to work on yet.")

    store.update(job_id, stage="finalising")
    source = _load(job_id, source_name)

    versions = list(state.get("versions") or [])
    step = len(versions) + 1

    if action == "upscale":
        result = _upscale(job_id, source, scale)
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


def _upscale(job_id: str, source: Image.Image, scale: float) -> Image.Image:
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

    if config.UPSCALE_DETAIL_PASS:
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

    src_dir, dest_dir = job_dir(job_id), job_dir(new_job_id)
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


# --- Runner ------------------------------------------------------------------

def run_auto(job_id: str, uploads: list[tuple[str, bytes]], note: str):
    """Ingest through drafts, then stop for a human to choose."""
    try:
        ingest(job_id, uploads, note)
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
                presets_wanted: list[str] | None = None):
    try:
        enhance(job_id, instruction, action, scale)
        if presets_wanted:
            export(job_id, presets_wanted)
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
