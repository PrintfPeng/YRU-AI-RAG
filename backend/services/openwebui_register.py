# backend/services/openwebui_register.py
"""
Self-registration: เมื่อ backend เริ่มทำงาน ให้ลงทะเบียนตัวเองกับ Open WebUI
เพื่อให้ model 'yru-rag-assistant' ปรากฏในรายการของ Open WebUI อัตโนมัติ

Flow:
  1. Login ด้วย admin credentials → JWT token
  2. GET /openai/config  → อ่าน URL list ปัจจุบัน
  3. ถ้า RAG_BACKEND_URL ยังไม่อยู่ใน list → POST /openai/config/update
  4. Log ผลลัพธ์
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

# ─── Config จาก Environment ────────────────────────────────────────────────────
OPEN_WEBUI_URL       = os.getenv("OPEN_WEBUI_URL", "").rstrip("/")
ADMIN_EMAIL          = os.getenv("OPEN_WEBUI_ADMIN_EMAIL", "")
ADMIN_PASSWORD       = os.getenv("OPEN_WEBUI_ADMIN_PASSWORD", "")
RAG_BACKEND_URL      = os.getenv("RAG_BACKEND_URL", "").rstrip("/")
RAG_BACKEND_KEY      = os.getenv("RAG_BACKEND_KEY", "yru-rag-key")

_TIMEOUT = 10  # วินาที


def _login(base: str, email: str, password: str) -> str | None:
    """Login และคืน JWT token"""
    try:
        r = requests.post(
            f"{base}/api/v1/auths/signin",
            json={"email": email, "password": password},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        token = r.json().get("token", "")
        if not token:
            logger.warning("[OWU-Register] Login succeeded but no token returned")
        return token
    except Exception as e:
        logger.warning(f"[OWU-Register] Login failed: {e}")
        return None


def _get_config(base: str, token: str) -> dict | None:
    """ดึง OpenAI config ปัจจุบันจาก Open WebUI"""
    try:
        r = requests.get(
            f"{base}/openai/config",
            headers={"Authorization": f"Bearer {token}"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"[OWU-Register] Get config failed: {e}")
        return None


def _update_config(base: str, token: str, config: dict) -> bool:
    """อัปเดต OpenAI config ใน Open WebUI"""
    try:
        r = requests.post(
            f"{base}/openai/config/update",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=config,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"[OWU-Register] Update config failed: {e}")
        return False


def register_with_openwebui() -> bool:
    """
    เพิ่ม RAG backend เข้า Open WebUI connection list (idempotent)
    คืน True ถ้าสำเร็จ หรือ connection อยู่แล้ว
    """
    # Validate ENV vars
    if not all([OPEN_WEBUI_URL, ADMIN_EMAIL, ADMIN_PASSWORD, RAG_BACKEND_URL]):
        logger.info(
            "[OWU-Register] Skipped — OPEN_WEBUI_URL / OPEN_WEBUI_ADMIN_EMAIL / "
            "OPEN_WEBUI_ADMIN_PASSWORD / RAG_BACKEND_URL not all set in environment"
        )
        return False

    logger.info(f"[OWU-Register] Registering with Open WebUI at {OPEN_WEBUI_URL} ...")

    # Step 1: Login
    token = _login(OPEN_WEBUI_URL, ADMIN_EMAIL, ADMIN_PASSWORD)
    if not token:
        return False

    # Step 2: Get current config
    cfg = _get_config(OPEN_WEBUI_URL, token)
    if cfg is None:
        return False

    urls: list = cfg.get("OPENAI_API_BASE_URLS", [])
    keys: list = cfg.get("OPENAI_API_KEYS", [])

    # Normalize for comparison (strip trailing slashes)
    rag_url_norm = RAG_BACKEND_URL.rstrip("/")
    existing_norm = [u.rstrip("/") for u in urls]

    if rag_url_norm in existing_norm:
        logger.info(
            f"[OWU-Register] Already registered → "
            f"yru-rag-assistant visible at index {existing_norm.index(rag_url_norm)}"
        )
        return True

    # Step 3: Append our backend
    urls.append(RAG_BACKEND_URL)
    keys.append(RAG_BACKEND_KEY)

    new_cfg = {
        "ENABLE_OPENAI_API": cfg.get("ENABLE_OPENAI_API", True),
        "OPENAI_API_BASE_URLS": urls,
        "OPENAI_API_KEYS": keys,
        "OPENAI_API_CONFIGS": cfg.get("OPENAI_API_CONFIGS", {}),
    }

    ok = _update_config(OPEN_WEBUI_URL, token, new_cfg)
    if ok:
        logger.info(
            f"[OWU-Register] SUCCESS — 'yru-rag-assistant' registered at index {len(urls)-1} "
            f"→ URL: {RAG_BACKEND_URL}"
        )
    else:
        logger.warning("[OWU-Register] FAILED to update config")

    return ok
