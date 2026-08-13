"""
Seed / re-seed Vanna training data into ChromaDB.

USAGE (from inside backend container):
    docker exec hybrid_rag_backend python scripts/seed_vanna_training.py [--reset] [--verify]

MODES:
    (default)      Reset + fresh seed.  Idempotent result — safe to run repeatedly.
    --verify       Only print current training count; do not modify.
    --add-only     Append data without wiping first (dev only — can cause duplicates
                   since Vanna generates random UUIDs per train call).

The training data itself lives in backend/services/vanna_training_data.py
Edit that file, then run this script to apply changes.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

# Make backend/ importable when running from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

# Load .env for local dev (docker exec inside container already has env from compose)
try:
    from dotenv import load_dotenv
    for candidate in (".env.local", ".env"):
        if os.path.exists(candidate):
            load_dotenv(candidate)
            print(f"[seed] loaded {candidate}", flush=True)
            break
except ImportError:
    pass

from backend.services.vanna_sql import _get_vn, _seed_training  # noqa: E402
from backend.services.vanna_training_data import (  # noqa: E402
    CORE_TABLES, BUSINESS_DOCS, SQL_EXAMPLES,
)


def _reset(vn) -> int:
    """Remove all training data. Returns count removed."""
    df = vn.get_training_data()
    n = len(df)
    if n == 0:
        return 0
    for _, row in df.iterrows():
        try:
            vn.remove_training_data(id=row["id"])
        except Exception as e:
            print(f"  [warn] failed to remove {row['id']}: {e}", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="Reset training data first (default mode also does this).")
    ap.add_argument("--verify", action="store_true",
                    help="Print current count only; do not modify.")
    ap.add_argument("--add-only", action="store_true",
                    help="Append without reset (may create duplicates).")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[seed] Chroma path = {os.getenv('VANNA_CHROMA_PATH', '/app/vanna_chroma')}", flush=True)
    print(f"[seed] Ollama      = {os.getenv('OLLAMA_BASE_URL')}", flush=True)
    print(f"[seed] LLM model   = {os.getenv('LLM_MODEL', 'qwen2.5:7b')}", flush=True)

    vn = _get_vn()

    existing = len(vn.get_training_data())
    print(f"[seed] Existing training rows: {existing}", flush=True)

    if args.verify:
        print("[seed] --verify: no changes made.", flush=True)
        return 0

    if not args.add_only:
        removed = _reset(vn)
        if removed:
            print(f"[seed] Wiped {removed} existing rows.", flush=True)

    expected = len(CORE_TABLES) + len(BUSINESS_DOCS) + len(SQL_EXAMPLES)
    print(f"[seed] Seeding {len(CORE_TABLES)} tables + {len(BUSINESS_DOCS)} docs + {len(SQL_EXAMPLES)} examples "
          f"(~{expected} rows expected)", flush=True)

    _seed_training(vn)

    final = len(vn.get_training_data())
    delta = final - (0 if not args.add_only else existing)
    print(f"[seed] Done in {time.time()-t0:.1f}s. Total training rows: {final} (added +{delta})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
