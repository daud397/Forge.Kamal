"""Make an image sharper and bigger with a real super-resolution network.

Lanczos (what "Enlarge" used to do) makes a file bigger but softer: it can only
spread the pixels it has. A super-resolution network has learned what edges,
weave and print look like at higher resolution, so it rebuilds crisp edges
instead of stretching blurry ones.

The network is Real-ESRGAN's general model (realesr-general-x4v3, BSD-3-Clause),
converted to ONNX by make_sharpen_model.py and committed as sharpen_x4.onnx. It
runs on the server's CPU: free per image, no API call, no spend against the
daily cap, and - unlike a generative edit - it cannot redraw the product, add
objects or change a motif. It only restores detail that is already implied.
"""
from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image, ImageFilter

import config

MODEL_PATH = config.ROOT / "sharpen_x4.onnx"

_session = None
_session_lock = threading.Lock()
# One image at a time. Two concurrent 8K jobs on a small container is how it
# runs out of memory, and the network already uses every core on its own.
_run_lock = threading.Lock()


class SharpenError(RuntimeError):
    pass


def available() -> bool:
    if not MODEL_PATH.exists():
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _sess():
    global _session
    with _session_lock:
        if _session is None:
            if not available():
                raise SharpenError("The sharpening model is not installed on this server.")
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(str(MODEL_PATH), opts,
                                            providers=["CPUExecutionProvider"])
        return _session


def _x4(rgb: np.ndarray, tile: int = 256, pad: int = 12, progress=None) -> np.ndarray:
    """rgb float32 HxWx3 in [0,1] -> 4H x 4W x 3. Run in overlapping tiles so
    memory stays flat however big the photo is; the overlap is cropped away,
    so tile seams never show."""
    sess = _sess()
    h, w, _ = rgb.shape
    out = np.zeros((h * 4, w * 4, 3), dtype=np.float32)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))[None]
    tiles = [(y, x) for y in range(0, h, tile) for x in range(0, w, tile)]
    for n, (y, x) in enumerate(tiles):
        y1, x1 = min(y + tile, h), min(x + tile, w)
        py0, px0 = max(0, y - pad), max(0, x - pad)
        py1, px1 = min(h, y1 + pad), min(w, x1 + pad)
        res = sess.run(None, {"input": chw[:, :, py0:py1, px0:px1]})[0][0]
        oy, ox = (y - py0) * 4, (x - px0) * 4
        out[y * 4:y1 * 4, x * 4:x1 * 4] = res[:, oy:oy + (y1 - y) * 4,
                                             ox:ox + (x1 - x) * 4].transpose(1, 2, 0)
        if progress and (n % 8 == 0 or n == len(tiles) - 1):
            progress(n + 1, len(tiles))
    return np.clip(out, 0.0, 1.0)


def target_size(size: tuple[int, int], scale: float) -> tuple[int, int]:
    """The output size for a requested factor, never past UPSCALE_MAX_EDGE and
    never smaller than the input."""
    w, h = size
    k = max(1.0, min(scale, config.UPSCALE_MAX_EDGE / max(w, h)))
    return max(1, round(w * k)), max(1, round(h * k))


def sharpen(img: Image.Image, scale: float = 2.0, strength: float = 0.85,
            progress=None) -> Image.Image:
    """Sharpen, and enlarge by `scale` (1 = same size, just sharper).

    strength blends the network's result with a plain Lanczos resize: 1.0 is
    the network alone, lower keeps more of the photo's own grain. 0.85 keeps
    fabric texture from looking painted while still clearly sharper.
    """
    src = img.convert("RGBA")
    out_w, out_h = target_size(src.size, scale)

    # The network enlarges 4x. Feed it at most SHARPEN_MAX_INPUT on the long
    # edge: past that the output would be bigger than anything we deliver, and
    # the time grows with every pixel. A 4000px phone photo is shrunk to the
    # cap first, which also averages away its sensor noise before rebuilding.
    limit = config.SHARPEN_MAX_INPUT
    feed = src
    if max(src.size) > limit:
        k = limit / max(src.size)
        feed = src.resize((round(src.width * k), round(src.height * k)), Image.LANCZOS)
    # Never feed more than the output needs: input x4 should only just cover it.
    need = max(out_w, out_h) / 4
    if max(feed.size) > need * 1.25 and need >= 64:
        k = (need * 1.25) / max(feed.size)
        feed = feed.resize((max(1, round(feed.width * k)), max(1, round(feed.height * k))),
                           Image.LANCZOS)

    rgb = np.asarray(feed.convert("RGB"), dtype=np.float32) / 255.0
    with _run_lock:
        big = _x4(rgb, progress=progress)
    sr = Image.fromarray(np.rint(big * 255).astype(np.uint8), "RGB")
    sr = sr.resize((out_w, out_h), Image.LANCZOS) if sr.size != (out_w, out_h) else sr

    plain = src.convert("RGB").resize((out_w, out_h), Image.LANCZOS)
    s = float(min(1.0, max(0.0, strength)))
    if s < 1.0:
        sr = Image.blend(plain, sr, s)

    # A light final crispening on brightness only, so colour edges don't fringe.
    y, cb, cr = sr.convert("YCbCr").split()
    y = y.filter(ImageFilter.UnsharpMask(radius=1.2, percent=40, threshold=2))
    result = Image.merge("YCbCr", (y, cb, cr)).convert("RGB")

    alpha = src.getchannel("A")
    if alpha.getextrema()[0] < 255:
        result = result.convert("RGBA")
        result.putalpha(alpha.resize((out_w, out_h), Image.LANCZOS))
    return result
