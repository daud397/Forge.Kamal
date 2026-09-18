"""Password gate.

Deliberately minimal: one shared password, an HMAC-signed cookie, no user
accounts. This is a tool for one team, and a login form that nobody maintains is
worse than one with nothing to maintain.

What it is not: multi-user, role-aware, or auditable per person. If you ever need
to know which colleague ran which job, replace this rather than extending it.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

COOKIE = "forge_session"
LIFETIME = 60 * 60 * 24 * 14        # two weeks

# Brute-force damping. Not a lockout - just enough that guessing by script is
# pointless against a single shared password.
_failures: dict[str, list[float]] = {}
MAX_ATTEMPTS = 8
WINDOW = 600


def _secret() -> bytes:
    key = os.getenv("SESSION_SECRET")
    if not key:
        # Fine locally; on a server this means sessions die on every restart,
        # which is why the deploy script generates one.
        key = "dev-secret-not-for-production"
    return key.encode()


def password() -> str | None:
    return os.getenv("PORTAL_PASSWORD") or None


def required() -> bool:
    """No password set means no gate. Correct for localhost, wrong for a VPS."""
    return password() is not None


def _sign(payload: str) -> str:
    mac = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{mac}"


def _verify(token: str) -> bool:
    try:
        payload, mac = token.rsplit(".", 1)
        expires = int(payload.split(":")[1])
    except (ValueError, IndexError):
        return False

    expected = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected):
        return False
    return time.time() < expires


def issue() -> str:
    return _sign(f"{secrets.token_urlsafe(8)}:{int(time.time() + LIFETIME)}")


def authorised(request: Request) -> bool:
    if not required():
        return True
    token = request.cookies.get(COOKIE)
    return bool(token and _verify(token))


def check_rate(ip: str):
    now = time.time()
    recent = [t for t in _failures.get(ip, []) if now - t < WINDOW]
    _failures[ip] = recent
    if len(recent) >= MAX_ATTEMPTS:
        raise HTTPException(429, "Too many attempts. Wait ten minutes.")


def record_failure(ip: str):
    _failures.setdefault(ip, []).append(time.time())


def attempt(supplied: str, ip: str) -> bool:
    check_rate(ip)
    if hmac.compare_digest(supplied, password() or ""):
        _failures.pop(ip, None)
        return True
    record_failure(ip)
    return False


LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listing Forge</title>
<style>
:root{--bench:#2b2b2b;--panel:#333;--edge:#454545;--ink:#eaeae7;--muted:#a3a39f;--cyan:#17a2c9}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
 background:var(--bench);color:var(--ink);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
form{background:var(--panel);border:1px solid var(--edge);border-radius:4px;
 padding:26px;width:300px}
h1{font-size:16px;margin:0 0 4px}
p{margin:0 0 18px;color:var(--muted);font-size:13px}
input{width:100%;box-sizing:border-box;padding:9px;background:#3a3a3a;
 border:1px solid var(--edge);border-radius:3px;color:var(--ink);font-size:14px}
input:focus{outline:2px solid #0d7391;border-color:var(--cyan)}
button{width:100%;margin-top:10px;padding:9px;background:var(--cyan);color:#07222b;
 border:0;border-radius:3px;font:600 13px inherit;cursor:pointer}
button:hover{background:#1fb4dd}
.err{color:#cd5c4f;font-size:13px;margin:10px 0 0}
</style></head>
<body>
<form method="post" action="/login">
  <h1>Listing Forge</h1>
  <p>Enter the portal password.</p>
  <input type="password" name="password" autofocus autocomplete="current-password">
  <button type="submit">Enter</button>
  __ERROR__
</form>
</body></html>"""


def login_page(error: str = "") -> HTMLResponse:
    html = LOGIN_PAGE.replace(
        "__ERROR__", f'<p class="err">{error}</p>' if error else ""
    )
    return HTMLResponse(html, status_code=401 if error else 200)


def redirect_to_login() -> RedirectResponse:
    return RedirectResponse("/login", status_code=303)
