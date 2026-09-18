"""Central configuration. Everything tunable lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
RENDERS = DATA / "renders"
DRAFTS = DATA / "drafts"
FINALS = DATA / "finals"
EXPORTS = DATA / "exports"
DB_PATH = DATA / "forge.db"

for d in (UPLOADS, RENDERS, DRAFTS, FINALS, EXPORTS):
    d.mkdir(parents=True, exist_ok=True)

# Bump this whenever the code changes. /api/status reports it, so there is a
# way to confirm which build is actually running rather than inferring it from
# behaviour - which has already cost us a debugging session once.
BUILD = "2026-09-18g"

# --- Models -----------------------------------------------------------------
REASONING_MODEL = os.getenv("REASONING_MODEL", "gpt-6-astra")
DRAFT_MODEL = os.getenv("DRAFT_MODEL", "gpt-image-2.5-flare")
FINAL_MODEL = os.getenv("FINAL_MODEL", "gpt-image-2.5-sunburst")

REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")  # low|medium|high|xhigh|max

DRAFT_QUALITY = os.getenv("DRAFT_QUALITY", "medium")
FINAL_QUALITY = os.getenv("FINAL_QUALITY", "high")

DRAFT_COUNT = int(os.getenv("DRAFT_COUNT", "3"))

# Run the whole pipeline with synthetic images and no API key. Useful for
# clicking through the UI before you spend anything.
MOCK = os.getenv("MOCK", "").lower() in ("1", "true", "yes")

# GPT-Image-2.5 mask convention. OpenAI's images.edit treats TRANSPARENT pixels
# as the editable region and opaque pixels as preserved. Some third-party
# gateways invert this (white = replace). If your first masked edit comes back
# with the background untouched and the product mangled, flip this to false.
MASK_TRANSPARENT_IS_EDITABLE = os.getenv("MASK_TRANSPARENT_IS_EDITABLE", "true").lower() != "false"

# --- GPT-Image-2.5 output size constraints (from the API docs) ---------------
MAX_EDGE = 3840
EDGE_MULTIPLE = 16
MAX_ASPECT = 3.0
MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400
EXPERIMENTAL_PIXELS = 3_686_400  # 2560x1440; above this OpenAI marks output experimental

# Generation happens below the experimental threshold, then Lanczos carries it
# the rest of the way. Cheaper and more predictable than generating at 4K.
DRAFT_SIZE = (1024, 1024)
FINAL_SIZE = (1536, 1536)

# --- CAD rendering ----------------------------------------------------------
RENDER_SIZE = int(os.getenv("RENDER_SIZE", "1024"))
MAX_FACES = int(os.getenv("MAX_FACES", "80000"))

# Default three-quarter view plus two alternates, as (azimuth, elevation) degrees.
CAMERA_ANGLES = {
    "hero": (35.0, 20.0),
    "front": (0.0, 5.0),
    "top": (35.0, 62.0),
}

# Upscaling: a generative detail pass genuinely adds detail, but it is a
# generative step and can change what it sharpens. Off means Lanczos only,
# which enlarges honestly and invents nothing.
UPSCALE_DETAIL_PASS = os.getenv("UPSCALE_DETAIL_PASS", "").lower() in ("1", "true", "yes")

# --- QA tolerances ----------------------------------------------------------
DELTA_E_PASS = float(os.getenv("DELTA_E_PASS", "2.0"))     # imperceptible to most viewers
DELTA_E_WARN = float(os.getenv("DELTA_E_WARN", "5.0"))     # noticeable side by side
SSIM_PASS = float(os.getenv("SSIM_PASS", "0.97"))          # structure of the preserved region

# --- Deployment: auth and spend ---------------------------------------------
# PORTAL_PASSWORD unset means no login gate. Correct on localhost, dangerous on
# a public server - the deploy script refuses to finish without one.
PORTAL_PASSWORD = os.getenv("PORTAL_PASSWORD")
SESSION_SECRET = os.getenv("SESSION_SECRET")

# Estimates, not billed amounts. Check your OpenAI dashboard after a few real
# runs and correct these, or the cap protects you by the wrong margin.
COST_PER_IMAGE = float(os.getenv("COST_PER_IMAGE", "0.19"))
COST_PER_TEXT_CALL = float(os.getenv("COST_PER_TEXT_CALL", "0.04"))

# Hard ceiling per UTC day. 0 disables it.
DAILY_CAP = float(os.getenv("DAILY_CAP", "25"))

CAD_EXTS = {".stl", ".obj", ".ply", ".glb", ".gltf", ".off", ".3mf", ".step", ".stp", ".iges", ".igs"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
