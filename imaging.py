"""Deterministic image maths: colour difference, structure, masks, resizing.

The QA numbers here are computed, not judged. A vision model asked "is this
colour right?" agrees with you far too readily; CIEDE2000 does not.
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps
from scipy.ndimage import uniform_filter, binary_dilation

import config


# --- sRGB -> CIE Lab ---------------------------------------------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: float array in [0,1], shape (..., 3). Returns L*a*b* under D65."""
    lin = _srgb_to_linear(rgb)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = lin @ m.T
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz / white

    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000. Inputs shape (..., 3), returns shape (...)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7 + 1e-12)))

    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2

    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hbp = np.where(
        C1p * C2p == 0, hsum,
        np.where(hdiff <= 180, hsum / 2,
                 np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)),
    )

    T = (1
         - 0.17 * np.cos(np.radians(hbp - 30))
         + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6))
         - 0.20 * np.cos(np.radians(4 * hbp - 63)))

    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7 + 1e-12))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc

    return np.sqrt(
        (dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


# --- Structural similarity ---------------------------------------------------

def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, win: int = 7) -> float:
    """Mean SSIM over grayscale arrays in [0,1], optionally restricted to a mask."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a = uniform_filter(a, win)
    mu_b = uniform_filter(b, win)
    saa = uniform_filter(a * a, win) - mu_a ** 2
    sbb = uniform_filter(b * b, win) - mu_b ** 2
    sab = uniform_filter(a * b, win) - mu_a * mu_b

    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2)
    smap = num / np.maximum(den, 1e-12)

    if mask is not None and mask.any():
        return float(smap[mask].mean())
    return float(smap.mean())


# --- Opening uploads ---------------------------------------------------------

# Refuse decompression bombs well before they fill the container's memory.
Image.MAX_IMAGE_PIXELS = 120_000_000

# iPhones save HEIC by default. Without this an iPhone photo was "unsupported".
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:                     # optional: everything else still works
    pass


def open_upload(path) -> Image.Image:
    """Open a seller's file the way their phone and browser show it.

    Two things a bare convert("RGBA") gets wrong: EXIF rotation (portrait phone
    photos came out sideways, and a painted mask no longer lined up), and
    high-bit-depth files (a 16-bit PNG clipped almost entirely to white, a
    float TIFF went black).
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) or img
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        top = float(arr.max()) if arr.size else 0.0
        if img.mode == "F" and top <= 1.0:
            scale = 255.0
        elif top <= 255:
            scale = 1.0
        else:
            scale = 255.0 / (65535.0 if top <= 65535 else top)
        img = Image.fromarray(np.clip(arr * scale, 0, 255).astype(np.uint8), "L")
    return img.convert("RGBA")


# --- Masks -------------------------------------------------------------------

def resize_rgba(img: Image.Image, size: tuple[int, int],
                resample=Image.LANCZOS) -> Image.Image:
    """Resize with transparency without picking up a dark fringe.

    Resampling RGBA directly averages the colour of fully transparent pixels -
    usually black - into every edge pixel, which shows up as a grey halo once
    the product lands on a light background. Premultiplying by alpha first, then
    dividing back out, keeps the edge colour honest.
    """
    if img.width * img.height > 8_000_000:
        # A 48MP phone photo in float64 peaked near 4GB here. Pillow's own
        # premultiplied mode does the same job in 8 bits.
        return img.convert("RGBA").convert("RGBa").resize(size, resample).convert("RGBA")

    arr = np.asarray(img.convert("RGBA"), dtype=np.float64) / 255.0
    alpha = arr[..., 3:4]

    premul = Image.fromarray(np.rint(arr[..., :3] * alpha * 255).astype(np.uint8), "RGB")
    a_img = Image.fromarray(np.rint(alpha[..., 0] * 255).astype(np.uint8), "L")

    pm_r = np.asarray(premul.resize(size, resample), dtype=np.float64) / 255.0
    a_r = np.asarray(a_img.resize(size, resample), dtype=np.float64)[..., None] / 255.0

    rgb = np.divide(pm_r, np.maximum(a_r, 1e-6))
    rgb = np.where(a_r > 1e-6, rgb, 0.0)

    out = np.concatenate([np.clip(rgb, 0, 1), np.clip(a_r, 0, 1)], axis=-1)
    return Image.fromarray(np.rint(out * 255).astype(np.uint8), "RGBA")


def flatten(img: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """Drop alpha onto a solid colour instead of letting it become black."""
    rgba = img.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*background, 255))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


def mask_from_alpha(render_rgba: Image.Image, grow: int = 3) -> Image.Image:
    """Product silhouette from a CAD render's alpha channel.

    `grow` dilates the preserved region by a few pixels so the editing model
    doesn't nibble the product's contour when it paints the new background.
    """
    alpha = np.array(render_rgba.convert("RGBA"))[..., 3] > 0
    if grow > 0:
        alpha = binary_dilation(alpha, iterations=grow)
    return Image.fromarray(np.where(alpha, 255, 0).astype(np.uint8), mode="L")


def measurement_mask(product_mask: Image.Image, erode: int = 5) -> Image.Image:
    """Shrink the product mask before measuring.

    The composite feathers its seam and the mask itself was dilated to protect
    the contour. Measuring right up to the edge therefore samples a ring of
    blended pixels and reports a colour shift that isn't on the product. Erode
    past that ring and the numbers describe the product only.
    """
    keep = np.array(product_mask) > 127
    if erode > 0:
        keep = ~binary_dilation(~keep, iterations=erode)
    if not keep.any():                      # tiny products: fall back rather than divide by zero
        keep = np.array(product_mask) > 127
    return Image.fromarray(np.where(keep, 255, 0).astype(np.uint8), mode="L")


def edit_mask(product_mask: Image.Image, size: tuple[int, int]) -> bytes:
    """Build the RGBA mask PNG that images.edit expects.

    OpenAI's convention: transparent pixels mark the region the model may
    change. So alpha = 0 across the background, 255 over the product. Flip
    MASK_TRANSPARENT_IS_EDITABLE in config if your gateway inverts this.
    """
    m = product_mask.resize(size, Image.LANCZOS)
    keep = np.array(m) > 127

    if not config.MASK_TRANSPARENT_IS_EDITABLE:
        keep = ~keep

    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.where(keep, 255, 0).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def mask_from_strokes(png_bytes: bytes, size: tuple[int, int]) -> Image.Image:
    """Turn painted canvas strokes into a preserved-region mask.

    The user paints over the product (what must survive); everything unpainted
    becomes editable.
    """
    painted = Image.open(io.BytesIO(png_bytes)).convert("RGBA").resize(size, Image.LANCZOS)
    covered = np.array(painted)[..., 3] > 16
    return Image.fromarray(np.where(covered, 255, 0).astype(np.uint8), mode="L")


# --- Compositing -------------------------------------------------------------

def composite_preserve(original: Image.Image, edited: Image.Image,
                       product_mask: Image.Image, feather: float = 1.5) -> Image.Image:
    """Paste the original product pixels back over the edited frame.

    The docs are explicit that prompting alone will not hold a region
    pixel-identical across edits. This makes it structural: the model supplies
    the background, the source supplies the product, and a short feather hides
    the seam.

    The composite happens at whichever is larger, the source or the generated
    frame. It used to happen at the generated size, so a 4000px product photo
    was shrunk to the model's 1920px before its own pixels went back in - the
    product was preserved, but at half its resolution. Now the generated
    background is enlarged to the source and the product keeps every pixel it
    was photographed with.
    """
    if original.width * original.height > edited.width * edited.height:
        size = original.size
        edit = edited.convert("RGB").resize(size, Image.LANCZOS).convert("RGBA")
        src = original.convert("RGBA")
    else:
        size = edited.size
        edit = edited.convert("RGBA")
        src = resize_rgba(original, size)      # premultiplied: no black halo
    feather = feather * max(1.0, size[0] / edited.width)

    # Where the source has its own alpha (a CAD render), that alpha is a better
    # edge than any painted or dilated mask, so intersect the two: the mask says
    # which object to keep, the alpha says exactly where its edge falls. Done in
    # 8 bits: at 24MP a float copy of every plane cost gigabytes.
    keep = product_mask.convert("L").resize(size, Image.LANCZOS)
    src_alpha = src.getchannel("A")
    if src_alpha.getextrema()[0] < 255:
        keep = ImageChops.multiply(keep, src_alpha)

    if feather > 0:
        keep = keep.filter(ImageFilter.GaussianBlur(feather))

    out = edit.copy()
    out.paste(src, (0, 0), keep)
    return out.convert("RGB")


def pad_to_ratio(img: Image.Image, ratio: float, fill=(0, 0, 0, 0)) -> Image.Image:
    """Extend the canvas to a new shape without stretching anything.

    Asked for 1:1 from a 4:3 photo, the old path squashed the photo into the
    square. Padding keeps the product's proportions; the padding is transparent,
    which the edit treats as area to fill, so the model extends the scene.
    """
    w, h = img.size
    if abs(w / h - ratio) < 0.01:
        return img
    if w / h < ratio:
        nw, nh = round(h * ratio), h
    else:
        nw, nh = w, round(w / ratio)
    mode = "L" if img.mode == "L" else "RGBA"
    canvas = Image.new(mode, (nw, nh), fill if mode == "RGBA" else 0)
    src = img if img.mode == mode else img.convert(mode)
    canvas.paste(src, ((nw - w) // 2, (nh - h) // 2))
    return canvas


def fit_within(img: Image.Image, max_edge: int) -> Image.Image:
    """Shrink to max_edge on the long side, keeping the shape. Never enlarges."""
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    k = max_edge / max(w, h)
    size = (max(1, round(w * k)), max(1, round(h * k)))
    return resize_rgba(img, size) if img.mode == "RGBA" else img.resize(size, Image.LANCZOS)


# --- Sizing ------------------------------------------------------------------

RATIO_WORDS = {
    "square": 1.0, "1:1": 1.0, "1x1": 1.0,
    "4:5": 0.8, "5:4": 1.25, "3:4": 0.75, "4:3": 4 / 3,
    "2:3": 2 / 3, "3:2": 1.5, "9:16": 0.5625, "16:9": 16 / 9,
    "portrait": 0.8, "landscape": 1.5, "vertical": 0.8, "horizontal": 1.5,
    "wide": 16 / 9,
}


def ratio_from_text(text: str) -> float | None:
    """Pull a requested aspect ratio out of what the person typed.

    They ask for "1:1 aspect ratio" in plain words and it is a hard requirement
    for a marketplace listing, not a stylistic hint, so it has to survive into
    the API call rather than being left to the model to honour or ignore.
    """
    import re
    t = (text or "").lower()

    m = re.search(r"\b(\d{1,2})\s*[:x]\s*(\d{1,2})\b", t)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        if w and h and 0.25 <= w / h <= 4:
            return w / h

    for word, r in RATIO_WORDS.items():
        if word in t:
            return r
    return None


def size_for_ratio(ratio: float, target_pixels: int = 1_500_000) -> tuple[int, int]:
    """A legal generation size at the requested shape."""
    import math
    h = math.sqrt(target_pixels / ratio)
    m = config.EDGE_MULTIPLE
    # Round down, so the pixel budget is a ceiling: rounding to nearest put a
    # 3:4 final at 3,700,736 pixels, over the experimental threshold it exists
    # to stay under.
    return snap_size(int(h * ratio) // m * m, int(h) // m * m)


def snap_size(width: int, height: int) -> tuple[int, int]:
    """Coerce a requested size into something the API will accept."""
    def snap(v):
        return max(config.EDGE_MULTIPLE,
                   min(config.MAX_EDGE, int(round(v / config.EDGE_MULTIPLE)) * config.EDGE_MULTIPLE))

    w, h = snap(width), snap(height)

    long_e, short_e = max(w, h), min(w, h)
    if long_e / short_e > config.MAX_ASPECT:
        short_e = snap(long_e / config.MAX_ASPECT)
        w, h = (long_e, short_e) if w >= h else (short_e, long_e)

    total = w * h
    if total < config.MIN_PIXELS:
        k = math.sqrt(config.MIN_PIXELS / total)
        w, h = snap(w * k), snap(h * k)
    elif total > config.MAX_PIXELS:
        # Round down here: rounding to nearest could land a few pixels over
        # the ceiling and the API refuses the whole call.
        k = math.sqrt(config.MAX_PIXELS / total)
        m = config.EDGE_MULTIPLE
        w = max(m, int(w * k) // m * m)
        h = max(m, int(h * k) // m * m)

    return w, h


def size_string(width: int, height: int) -> str:
    w, h = snap_size(width, height)
    return f"{w}x{h}"


def upscale(img: Image.Image, target: tuple[int, int], fit: str = "contain",
            background: str = "white") -> Image.Image:
    """Lanczos only. No second generative pass, so nothing new is invented."""
    tw, th = target
    src = img.convert("RGBA")
    sw, sh = src.size

    k = min(tw / sw, th / sh) if fit == "contain" else max(tw / sw, th / sh)
    resized = resize_rgba(src, (max(1, round(sw * k)), max(1, round(sh * k))))

    if background == "transparent":
        canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    else:
        canvas = Image.new("RGBA", (tw, th), (255, 255, 255, 255))

    canvas.paste(resized, ((tw - resized.width) // 2, (th - resized.height) // 2), resized)

    if fit == "cover":
        left, top = (resized.width - tw) // 2, (resized.height - th) // 2
        canvas = resized.crop((max(0, left), max(0, top),
                               max(0, left) + tw, max(0, top) + th))
    return canvas


def region_stats(original: Image.Image, result: Image.Image,
                 product_mask: Image.Image) -> dict:
    """Colour and structure fidelity of the product region, plus the edited area."""
    # Measured at no more than 1536px on the long edge. Full-resolution
    # composites are now 20MP and more, and CIEDE2000 in float64 over that many
    # pixels costs gigabytes for a number that does not change.
    size = fit_within(result, 1536).size
    orig = np.asarray(flatten(resize_rgba(original, size)), dtype=np.float64) / 255.0
    res = np.asarray(flatten(resize_rgba(result.convert("RGBA"), size)),
                     dtype=np.float64) / 255.0

    m = np.asarray(product_mask.resize(size, Image.LANCZOS)) > 127
    if not m.any():
        m = np.ones(size[::-1], dtype=bool)

    lab_o, lab_r = rgb_to_lab(orig), rgb_to_lab(res)
    de = delta_e_2000(lab_o, lab_r)

    gray_o = orig @ np.array([0.2126, 0.7152, 0.0722])
    gray_r = res @ np.array([0.2126, 0.7152, 0.0722])

    de_masked = de[m]
    return {
        "delta_e_mean": round(float(de_masked.mean()), 3),
        "delta_e_p95": round(float(np.percentile(de_masked, 95)), 3),
        "delta_e_max": round(float(de_masked.max()), 3),
        "ssim_product": round(ssim(gray_o, gray_r, mask=m), 4),
        "product_coverage": round(float(m.mean()), 4),
        "pixels_out_of_tolerance": int((de_masked > config.DELTA_E_WARN).sum()),
    }


def verdict(stats: dict) -> tuple[str, list[str]]:
    """Traffic light from the measured numbers alone."""
    notes: list[str] = []
    level = "pass"

    if stats["delta_e_mean"] > config.DELTA_E_WARN:
        level = "fail"
        notes.append(
            f"Average colour shift of {stats['delta_e_mean']} dE2000 across the product "
            f"is above the {config.DELTA_E_WARN} limit - visibly the wrong colour."
        )
    elif stats["delta_e_mean"] > config.DELTA_E_PASS:
        level = "warn"
        notes.append(
            f"Colour drifted {stats['delta_e_mean']} dE2000 on average. Fine alone, "
            "detectable next to the real product."
        )

    if stats["ssim_product"] < config.SSIM_PASS:
        level = "fail"
        notes.append(
            f"Product structure similarity {stats['ssim_product']} is below "
            f"{config.SSIM_PASS}. Geometry or surface pattern changed during the edit."
        )

    if not notes:
        notes.append("Colour and geometry held within tolerance.")
    return level, notes
