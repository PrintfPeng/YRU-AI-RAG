#!/usr/bin/env python3
"""
scripts/register_openwebui.py
────────────────────────────────────────────────────────────────────
ลงทะเบียน YRU RAG Backend เข้า Open WebUI (เรียกใช้แบบ standalone)

วิธีใช้:
    python scripts/register_openwebui.py

หรือกำหนด ENV ก่อน:
    OPEN_WEBUI_URL=http://10.20.41.229:3000 \
    OPEN_WEBUI_ADMIN_EMAIL=admin@example.com \
    OPEN_WEBUI_ADMIN_PASSWORD=yourpassword \
    RAG_BACKEND_URL=http://10.20.41.108:8005/v1 \
    python scripts/register_openwebui.py
"""

import os, sys, json, requests
from pathlib import Path

# ─── Load .env if present ────────────────────────────────────────────────────
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# ─── Config ──────────────────────────────────────────────────────────────────
OPEN_WEBUI_URL    = os.getenv("OPEN_WEBUI_URL", "http://10.20.41.229:3000").rstrip("/")
ADMIN_EMAIL       = os.getenv("OPEN_WEBUI_ADMIN_EMAIL", "printfpeng@gmail.com")
ADMIN_PASSWORD    = os.getenv("OPEN_WEBUI_ADMIN_PASSWORD", "19735500")
RAG_BACKEND_URL   = os.getenv("RAG_BACKEND_URL", "http://10.20.41.108:8005/v1").rstrip("/")
RAG_BACKEND_KEY   = os.getenv("RAG_BACKEND_KEY", "yru-rag-key")
TIMEOUT = 10

def step(msg): print(f"  {msg}")
def ok(msg):   print(f"  ✅ {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def err(msg):  print(f"  ❌ {msg}"); sys.exit(1)

print()
print("=" * 60)
print("  YRU RAG → Open WebUI Self-Registration")
print("=" * 60)
print(f"  Open WebUI : {OPEN_WEBUI_URL}")
print(f"  Admin      : {ADMIN_EMAIL}")
print(f"  RAG URL    : {RAG_BACKEND_URL}")
print()

# Step 1: Verify backend is alive
step("Checking RAG backend ...")
try:
    r = requests.get(f"{RAG_BACKEND_URL}/models", timeout=TIMEOUT)
    models = r.json().get("data", [])
    ids = [m["id"] for m in models]
    ok(f"Backend OK — models: {ids}")
except Exception as e:
    err(f"Cannot reach RAG backend at {RAG_BACKEND_URL}: {e}")

# Step 2: Login to Open WebUI
step("Logging in to Open WebUI ...")
try:
    r = requests.post(
        f"{OPEN_WEBUI_URL}/api/v1/auths/signin",
        json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    token = r.json().get("token", "")
    role  = r.json().get("role", "")
    if not token:
        err("Login OK but no token returned")
    ok(f"Logged in as {ADMIN_EMAIL} (role={role})")
except Exception as e:
    err(f"Login failed: {e}")

HDR = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Step 3: Get current OpenAI config
step("Reading current OpenAI connections ...")
try:
    r = requests.get(f"{OPEN_WEBUI_URL}/openai/config", headers=HDR, timeout=TIMEOUT)
    r.raise_for_status()
    cfg = r.json()
    urls = cfg.get("OPENAI_API_BASE_URLS", [])
    keys = cfg.get("OPENAI_API_KEYS", [])
    print(f"      Current URLs: {urls}")
except Exception as e:
    err(f"Cannot read config: {e}")

# Step 4: Check if already registered
norm_urls = [u.rstrip("/") for u in urls]
if RAG_BACKEND_URL in norm_urls:
    idx = norm_urls.index(RAG_BACKEND_URL)
    ok(f"Already registered at index {idx} — nothing to do")
    print()
    print("  Run the following to verify the model is visible:")
    print(f"    curl -H 'Authorization: Bearer {token[:20]}...' "
          f"{OPEN_WEBUI_URL}/api/models | python -m json.tool")
    print()
    sys.exit(0)

# Step 5: Append our backend
step("Adding RAG backend connection ...")
urls.append(RAG_BACKEND_URL)
keys.append(RAG_BACKEND_KEY)

new_cfg = {
    "ENABLE_OPENAI_API": cfg.get("ENABLE_OPENAI_API", True),
    "OPENAI_API_BASE_URLS": urls,
    "OPENAI_API_KEYS": keys,
    "OPENAI_API_CONFIGS": cfg.get("OPENAI_API_CONFIGS", {}),
}

try:
    r = requests.post(
        f"{OPEN_WEBUI_URL}/openai/config/update",
        headers=HDR,
        json=new_cfg,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    saved = r.json()
    ok(f"Config updated — URLs: {saved.get('OPENAI_API_BASE_URLS')}")
except Exception as e:
    err(f"Config update failed: {e}")

# Step 6: Verify model appears
import time
step("Verifying model appears in Open WebUI ...")
time.sleep(2)
try:
    r = requests.get(
        f"{OPEN_WEBUI_URL}/api/models",
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )
    models = r.json().get("data", [])
    ids = [m["id"] for m in models]
    if "yru-rag-assistant" in ids:
        idx = next(m.get("urlIdx") for m in models if m["id"] == "yru-rag-assistant")
        ok(f"'yru-rag-assistant' visible in Open WebUI (urlIdx={idx})")
    else:
        warn(f"Model not visible yet. Models found: {ids}")
except Exception as e:
    warn(f"Verification check failed: {e}")

print()
print("=" * 60)
print("  DONE — Open http://10.20.41.229:3000/ and select")
print("         'YRU RAG Assistant' from the model dropdown")
print("=" * 60)
print()
