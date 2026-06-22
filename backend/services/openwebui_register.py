# backend/services/openwebui_register.py
"""
Self-registration: เมื่อ backend เริ่มทำงาน ให้ลงทะเบียนตัวเองกับ Open WebUI
"""
import os, logging, requests
logger = logging.getLogger(__name__)

OPEN_WEBUI_URL  = os.getenv("OPEN_WEBUI_URL", "").rstrip("/")
ADMIN_EMAIL     = os.getenv("OPEN_WEBUI_ADMIN_EMAIL", "")
ADMIN_PASSWORD  = os.getenv("OPEN_WEBUI_ADMIN_PASSWORD", "")
RAG_BACKEND_URL = os.getenv("RAG_BACKEND_URL", "").rstrip("/")
RAG_BACKEND_KEY = os.getenv("RAG_BACKEND_KEY", "yru-rag-key")
_TIMEOUT = 10

def _login(base, email, password):
    try:
        r = requests.post(f"{base}/api/v1/auths/signin",
                          json={"email": email, "password": password}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json().get("token", "")
    except Exception as e:
        logger.warning(f"[OWU-Register] Login failed: {e}")
        return None

def _get_config(base, token):
    try:
        r = requests.get(f"{base}/openai/config",
                         headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"[OWU-Register] Get config failed: {e}")
        return None

def _update_config(base, token, config):
    try:
        r = requests.post(f"{base}/openai/config/update",
                          headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                          json=config, timeout=_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"[OWU-Register] Update config failed: {e}")
        return False

def register_with_openwebui():
    if not all([OPEN_WEBUI_URL, ADMIN_EMAIL, ADMIN_PASSWORD, RAG_BACKEND_URL]):
        logger.info("[OWU-Register] Skipped - ENV vars not set")
        return False
    logger.info(f"[OWU-Register] Registering with Open WebUI at {OPEN_WEBUI_URL} ...")
    token = _login(OPEN_WEBUI_URL, ADMIN_EMAIL, ADMIN_PASSWORD)
    if not token:
        return False
    cfg = _get_config(OPEN_WEBUI_URL, token)
    if cfg is None:
        return False
    urls = cfg.get("OPENAI_API_BASE_URLS", [])
    keys = cfg.get("OPENAI_API_KEYS", [])
    if RAG_BACKEND_URL.rstrip("/") in [u.rstrip("/") for u in urls]:
        logger.info(f"[OWU-Register] Already registered")
        return True
    urls.append(RAG_BACKEND_URL)
    keys.append(RAG_BACKEND_KEY)
    ok = _update_config(OPEN_WEBUI_URL, token, {
        "ENABLE_OPENAI_API": cfg.get("ENABLE_OPENAI_API", True),
        "OPENAI_API_BASE_URLS": urls,
        "OPENAI_API_KEYS": keys,
        "OPENAI_API_CONFIGS": cfg.get("OPENAI_API_CONFIGS", {}),
    })
    if ok:
        logger.info(f"[OWU-Register] SUCCESS - yru-rag-assistant registered -> {RAG_BACKEND_URL}")
    else:
        logger.warning("[OWU-Register] FAILED to update config")
    return ok
