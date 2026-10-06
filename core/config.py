import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

PROFILES_DIR = BASE_DIR / "data" / "profiles"
STTM_DIR = BASE_DIR / "data" / "sttm"
BRONZE_DIR = BASE_DIR / "data" / "bronze_layer"
SILVER_DIR = BASE_DIR / "data" / "silver_layer"
GOLD_DIR = BASE_DIR / "data" / "gold_layer"
REPORTS_DIR = BASE_DIR / "data" / "reports"
AUDIT_DIR = BASE_DIR / "data" / "audit_logs"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

def ensure_dirs() -> None:
    for d in (
        PROFILES_DIR, STTM_DIR, BRONZE_DIR, SILVER_DIR,
        GOLD_DIR, REPORTS_DIR, AUDIT_DIR, UPLOADS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)