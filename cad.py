"""Turn a CAD file into raster renders plus a pixel-exact alpha matte.

Why a hand-rolled rasterizer instead of Blender or pyrender: both need a GPU
context or OSMesa, which turns "clone and run" into an afternoon of driver work.
This is pure numpy, runs anywhere Python runs, and is deterministic - the same
file always produces the same pixels, which matters when you diff renders during
QA.

The alpha channel is the real prize. Segmenting a product out of a photograph is
guesswork; here the silhouette comes straight out of the depth buffer, so the
mask handed to the editing model is exact.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

import config


class CADError(RuntimeError):
    pass


def load_mesh(path: Path):
    """Load a mesh, concatenating multi-part scenes into one body."""
    import trimesh

    suffix = path.suffix.lower()

    if suffix in (".step", ".stp", ".iges", ".igs"):
        try:
            import importlib
            importlib.import_module("cascadio")
        except ImportError as exc:
            raise CADError(
                "STEP and IGES need the OpenCascade bridge. Install it with "
                "`pip install cascadio`, or export your model as STL/OBJ/GLB."
            ) from exc

    try:
        loaded = trimesh.load(str(path), force="mesh")
    except Exception as exc:
        raise CADError(f"Could not read {path.name}: {exc}") from exc

    if hasattr(loaded, "geometry"):
        parts = list(loaded.geometry.values())
        if not parts:
            raise CADError(f"{path.name} contains no geometry.")
        loaded = trimesh.util.concatenate(parts)

    if not hasattr(loaded, "faces") or len(loaded.faces) == 0:
        raise CADError(f"{path.name} has no triangles. Point clouds are not supported.")

    if len(loaded.faces) > config.MAX_FACES:
        loaded = loaded.simplify_quadric_decimation(config.MAX_FACES)

    return loaded


def _normalise(vertices: np.ndarray) -> np.ndarray:
    """Centre on the origin and scale the longest axis to 1."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    centre = (lo + hi) / 2.0
    scale = float((hi - lo).max()) or 1.0
    return (vertices - centre) / scale


def _view_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    ry = np.array([
        [math.cos(az), 0.0, math.sin(az)],
        [0.0, 1.0, 0.0],
        [-math.sin(az), 0.0, math.cos(az)],
    ])
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(el), -math.sin(el)],
        [0.0, math.sin(el), math.cos(el)],
    ])
    return rx @ ry


def render(
    mesh,
    size: int = None,
    azimuth: float = 35.0,
    elevation: float = 20.0,
    base_colour: tuple[int, int, int] = (176, 178, 182),
    margin: float = 0.88,
) -> Image.Image:
    """Rasterise the mesh to an RGBA image. Transparent where the product isn't.

    Three-point studio lighting approximated per-face: a key from upper left, a
    cooler fill from the right, and a rim from behind to separate the silhouette
    from whatever background gets generated later.
    """
    size = size or config.RENDER_SIZE

    verts = _normalise(np.asarray(mesh.vertices, dtype=np.float64))
    faces = np.asarray(mesh.faces, dtype=np.int64)

    rot = _view_matrix(azimuth, elevation)
    vcam = verts @ rot.T

    # Fit the rotated bounds into the frame so no angle clips.
    lo, hi = vcam[:, :2].min(axis=0), vcam[:, :2].max(axis=0)
    span = float((hi - lo).max()) or 1.0
    scale = (size * margin) / span
    centre2d = (lo + hi) / 2.0

    sx = (vcam[:, 0] - centre2d[0]) * scale + size / 2.0
    sy = (-(vcam[:, 1] - centre2d[1])) * scale + size / 2.0
    depth = vcam[:, 2]

    tri = np.stack([sx, sy], axis=1)[faces]           # (F, 3, 2)
    tri_depth = depth[faces]                           # (F, 3)

    # Winding order in screen space tells us which faces point away.
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    area2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    front = area2 < -1e-9
    if not front.any():                                # flipped normals in the file
        front = area2 > 1e-9
        area2 = -area2
        tri = tri[:, ::-1, :]
        tri_depth = tri_depth[:, ::-1]

    tri, tri_depth, area2 = tri[front], tri_depth[front], area2[front]
    if len(tri) == 0:
        raise CADError("Nothing visible from this camera angle.")

    normals = np.asarray(mesh.face_normals, dtype=np.float64)[front] @ rot.T

    key = np.array([-0.45, 0.75, 0.50]); key /= np.linalg.norm(key)
    fill = np.array([0.80, 0.15, 0.45]); fill /= np.linalg.norm(fill)
    rim = np.array([0.10, 0.30, -0.95]); rim /= np.linalg.norm(rim)

    lam = (
        0.62 * np.clip(normals @ key, 0, None)
        + 0.24 * np.clip(normals @ fill, 0, None)
        + 0.30 * np.clip(normals @ rim, 0, None) ** 3
        + 0.20
    )
    shade = np.clip(lam, 0.0, 1.35)

    base = np.array(base_colour, dtype=np.float64) / 255.0
    face_rgb = np.clip(base[None, :] * shade[:, None], 0.0, 1.0)

    colour = np.zeros((size, size, 3), dtype=np.float64)
    zbuf = np.full((size, size), np.inf)
    covered = np.zeros((size, size), dtype=bool)

    # Painter-free: z-buffer per triangle, bounding box vectorised.
    order = np.argsort(tri_depth.min(axis=1))
    for idx in order:
        t = tri[idx]
        x0 = max(int(np.floor(t[:, 0].min())), 0)
        x1 = min(int(np.ceil(t[:, 0].max())) + 1, size)
        y0 = max(int(np.floor(t[:, 1].min())), 0)
        y1 = min(int(np.ceil(t[:, 1].max())) + 1, size)
        if x1 <= x0 or y1 <= y0:
            continue

        ys, xs = np.mgrid[y0:y1, x0:x1]
        px = xs + 0.5
        py = ys + 0.5

        denom = area2[idx]
        if abs(denom) < 1e-12:
            continue

        w0 = ((t[1, 0] - t[0, 0]) * (py - t[0, 1]) - (t[1, 1] - t[0, 1]) * (px - t[0, 0])) / denom
        w1 = ((t[2, 0] - t[1, 0]) * (py - t[1, 1]) - (t[2, 1] - t[1, 1]) * (px - t[1, 0])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue

        d = w1 * tri_depth[idx, 0] + w2 * tri_depth[idx, 1] + w0 * tri_depth[idx, 2]
        window = zbuf[y0:y1, x0:x1]
        nearer = inside & (d < window)
        if not nearer.any():
            continue

        window[nearer] = d[nearer]
        colour[y0:y1, x0:x1][nearer] = face_rgb[idx]
        covered[y0:y1, x0:x1][nearer] = True

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(colour * 255).astype(np.uint8)
    rgba[..., 3] = np.where(covered, 255, 0).astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGBA")
    return _antialias(img)


def _antialias(img: Image.Image) -> Image.Image:
    """Edge softening via a 2x round trip.

    The downsample has to be premultiplied, or every silhouette pixel averages
    in the transparent black of the background and the product ships with a
    grey outline around it.
    """
    from imaging import resize_rgba

    w, h = img.size
    big = img.resize((w * 2, h * 2), Image.NEAREST)
    return resize_rgba(big, (w, h))


def render_angles(path: Path, job_dir: Path) -> dict[str, str]:
    """Render the configured camera angles and write them alongside each other."""
    mesh = load_mesh(path)
    out: dict[str, str] = {}
    for name, (az, el) in config.CAMERA_ANGLES.items():
        img = render(mesh, azimuth=az, elevation=el)
        dest = job_dir / f"render_{name}.png"
        img.save(dest)
        out[name] = dest.name
    return out
