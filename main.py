"""HTTP layer. Thin on purpose - all the work is in pipeline.py."""
from __future__ import annotations

import base64
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

import agent, auth, budget, config, models, pipeline, store
from presets import PRESETS

app = FastAPI(title="Listing Forge")
pool = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
store.init()
budget.init()

OPEN_PATHS = {"/login", "/favicon.ico"}


@app.middleware("http")
async def gate(request: Request, call_next):
    """Everything behind the password, if one is set."""
    path = request.url.path
    if path in OPEN_PATHS or auth.authorised(request):
        return await call_next(request)

    if path.startswith("/api/"):
        return JSONResponse({"error": "Not signed in."}, status_code=401)
    return auth.redirect_to_login()


@app.get("/login")
def login_form(request: Request):
    if auth.authorised(request):
        return RedirectResponse("/", status_code=303)
    return auth.login_page()


@app.post("/login")
def login_submit(request: Request, password: str = Form("")):
    ip = request.client.host if request.client else "unknown"
    try:
        ok = auth.attempt(password, ip)
    except HTTPException as exc:
        return auth.login_page(exc.detail)

    if not ok:
        return auth.login_page("Wrong password.")

    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        auth.COOKIE, auth.issue(),
        max_age=auth.LIFETIME, httponly=True, samesite="lax",
        secure=os.getenv("HTTPS", "").lower() in ("1", "true", "yes"),
    )
    return response


@app.post("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(auth.COOKIE)
    return response


@app.get("/api/spend")
def spend():
    return budget.summary()



@app.get("/api/status")
def status():
    return {
        "mode": "live" if models.live() else "mock",
        "reasoning_model": config.REASONING_MODEL,
        "draft_model": config.DRAFT_MODEL,
        "final_model": config.FINAL_MODEL,
        "draft_size": f"{config.DRAFT_SIZE[0]}x{config.DRAFT_SIZE[1]}",
        "final_size": f"{config.FINAL_SIZE[0]}x{config.FINAL_SIZE[1]}",
        "tolerances": {"delta_e_pass": config.DELTA_E_PASS,
                       "delta_e_warn": config.DELTA_E_WARN,
                       "ssim_pass": config.SSIM_PASS},
        "auth": auth.required(),
        "spend": budget.summary(),
        "presets": {k: {"label": v["label"],
                        "size": f"{v['size'][0]}x{v['size'][1]}",
                        "format": v["format"],
                        "generative": v["generative"],
                        "note": v["note"]} for k, v in PRESETS.items()},
    }


@app.post("/api/jobs")
async def create_job(files: list[UploadFile] = File(...), note: str = Form("")):
    uploads = []
    for f in files:
        blob = await f.read()
        if len(blob) > 80 * 1024 * 1024:
            raise HTTPException(413, f"{f.filename} is over 80 MB.")
        uploads.append((f.filename, blob))

    if not uploads:
        raise HTTPException(400, "No files received.")

    job_id = store.create({"created_at": None})
    pool.submit(pipeline.run_auto, job_id, uploads, note)
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": store.recent()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    state = store.get(job_id)
    if not state:
        raise HTTPException(404, "No such job.")
    return state


@app.post("/api/jobs/{job_id}/finalise")
async def finalise(job_id: str,
                   draft_index: int = Form(...),
                   instruction: str = Form(""),
                   presets: str = Form(""),
                   mask: UploadFile | None = File(None)):
    if not store.get(job_id):
        raise HTTPException(404, "No such job.")

    painted = await mask.read() if mask else None
    wanted = [p for p in presets.split(",") if p.strip() in PRESETS]

    pool.submit(pipeline.run_finalise, job_id, draft_index, instruction, painted, wanted)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/export")
def export(job_id: str, presets: str = Form(...)):
    wanted = [p for p in presets.split(",") if p.strip() in PRESETS]
    if not wanted:
        raise HTTPException(400, "No valid presets named.")
    pool.submit(pipeline.export, job_id, wanted)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/chat")
def chat(job_id: str, message: str = Form(...)):
    if not store.get(job_id):
        raise HTTPException(404, "No such job.")
    if not message.strip():
        raise HTTPException(400, "Empty message.")
    try:
        return agent.respond(job_id, message.strip(), pool)
    except Exception as exc:
        return {"reply": f"That failed: {exc}", "actions": []}


@app.get("/api/jobs/{job_id}/file/{name}")
def job_file(job_id: str, name: str):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "Bad filename.")
    path = pipeline.job_dir(job_id) / name
    if not path.exists():
        raise HTTPException(404, "No such file.")
    return FileResponse(path)


@app.exception_handler(Exception)
def unhandled(request, exc):
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


# The three front-end files are served by name rather than by mounting this
# directory. A static mount here would sit on top of the source tree and hand
# out auth.py and config.py to anyone who guessed the filename.
FRONTEND = {
    "index.html": "text/html; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}


@app.get("/")
def index():
    return FileResponse(config.ROOT / "index.html", media_type=FRONTEND["index.html"])


@app.get("/{filename}")
def frontend(filename: str):
    media_type = FRONTEND.get(filename)
    if not media_type:
        raise HTTPException(404, "Not found.")
    return FileResponse(config.ROOT / filename, media_type=media_type)
