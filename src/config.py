import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root
load_dotenv()

def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

# Public config values used by the app
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = _require("OPENAI_API_KEY")   # ✅ correct
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

ENV = os.getenv("ENV", "dev")
PORT = int(os.getenv("PORT", "8000"))

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./var/index")).resolve()

# Create folders if missing
INDEX_DIR.mkdir(parents=True, exist_ok=True)