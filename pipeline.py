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
    return store.update(job_id, stage="drafted", drafts=out)


# --- Stage 4: final masked edit + QA ----------------------------------------

def finalise(job_id: str, draft_index: int, extra_instruction: str = "",
             painted_mask: bytes | None = None) -> dict:
    state = store.get(job_id)
    store.update(job_id, stage="finalising")

    chosen = next((d for d in state.get("drafts", []) if d["index"] == draft_index), None)
    if not chosen:
        raise ValueError(f"No draft with index {draft_index}.")

    base = _load(job_id, state["base_image"])
    size = config.FINAL_SIZE

    # Where does the product mask come from?
    if painted_mask:
        product_mask = imaging.mask_from_strokes(painted_mask, base.size)
        store.log(job_id, "Using the painted mask as the preserved region.")
    elif state.get("mask_source") == "cad_alpha":
        product_mask = imaging.mask_from_alpha(base)
        store.log(job_id, "Preserved region taken from the CAD alpha matte - pixel exact, no segmentation.")
    else:
        product_mask = None
        store.log(job_id, "No mask available. The edit runs unmasked and the product is not protected.", "warn")

    mask_png = imaging.edit_mask(product_mask, size) if product_mask else None

    prompt = chosen["prompt"]
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
    level, notes = imaging.verdict(stats)
    store.log(job_id, f"Model output: dE2000 mean {stats['delta_e_mean']}, "
                      f"p95 {stats['delta_e_p95']}, SSIM {stats['ssim_product']} -> {level}.")

    # What survives after compositing. Should be near zero; if it isn't, the
    # mask and the render have drifted out of alignment.
    residual = imaging.region_stats(base, composited, measure_mask)
    store.log(job_id, f"After composite: dE2000 mean {residual['delta_e_mean']}, "
                      f"SSIM {residual['ssim_product']}.")

    if level != "pass" and residual["delta_e_mean"] < config.DELTA_E_PASS:
        notes.append(
            "The composite corrected this: delivered product pixels come from your "
            "source file, so the shipped image is clean. The drift is a signal that "
            "the prompt or mask needs work, not a reason to reject this export."
        )

    # Judged QA.
    review = models.qa_review(composited, state.get("brief") or {})
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
