import os
from pathlib import Path
from dotenv import load_dotenv

# โหลด .env ถ้ามี
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# --- Custom API (OpenAI Compatible) ---

CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
CUSTOM_API_BASE = os.getenv("CUSTOM_API_BASE")
# ตั้งค่าโมเดลเริ่มต้นเป็น qwen/qwen-2.5-72b-instruct
CUSTOM_MODEL_NAME = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- OCR API Configuration ---
# URL จาก Swagger
OCR_API_URL = os.getenv("OCR_API_URL", "https://111.223.37.41:9001")

# Username/Password
OCR_USERNAME = os.getenv("OCR_USERNAME", "aiuser")
OCR_PASSWORD = os.getenv("OCR_PASSWORD", "aiuser@S0ftnix")

# SSL Verification Logic
# Default เป็น False ตามที่คุณต้องการ แต่ถ้าใน .env ส่งมาเป็น 'True' ก็จะเปิด verify ได้
_verify_ssl_env = os.getenv("VERIFY_SSL", "False").lower()
VERIFY_SSL = _verify_ssl_env in ("true", "1", "t")