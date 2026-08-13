"""
Vanna-based SQL agent — drop-in replacement for sql_agent.generate_and_run_sql().

Signature matches sql_agent.generate_and_run_sql(query: str) -> str
Returns a natural-language Thai summary (not dict) — identical shape to the legacy agent
so backend/main.py can swap via feature flag without changing downstream code.
"""
from __future__ import annotations
import os
import re
import time
from threading import Lock
from typing import Optional

from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.ollama import Ollama

# Reuse year handling + caching + OOD guard from legacy sql_agent (no need to reinvent)
from backend.services.sql_agent import (
    _normalize_years,
    _fix_sql_years,
    _get_cached,
    _set_cache,
    _is_ood,
)
from backend.services.vanna_training_data import (
    CORE_TABLES,
    BUSINESS_DOCS,
    SQL_EXAMPLES,
)


# ── Vanna class = ChromaDB (vector store) + Ollama (LLM) ──
class _YruVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config: dict):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


# ── Thread-safe lazy singleton (FastAPI async workers may call concurrently) ──
_vn: Optional[_YruVanna] = None
_vn_lock = Lock()


def _get_vn() -> _YruVanna:
    global _vn
    if _vn is not None:
        return _vn
    with _vn_lock:
        if _vn is None:
            _vn = _build_and_init()
    return _vn


def _build_and_init() -> _YruVanna:
    print("[vanna_sql] Building Vanna instance...", flush=True)
    vn = _YruVanna(config={
        "model": os.getenv("LLM_MODEL", "qwen2.5:7b"),
        "ollama_host": os.getenv("OLLAMA_BASE_URL", "http://172.19.0.1:11434"),
        "path": os.getenv("VANNA_CHROMA_PATH", "/app/vanna_chroma"),
        "options": {"temperature": 0.0, "num_predict": 512},
    })
    vn.connect_to_mysql(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", "3306")),
    )
    existing = len(vn.get_training_data())
    if existing == 0:
        print("[vanna_sql] Chroma empty — bootstrap-seeding baseline training", flush=True)
        _seed_training(vn)
        print(f"[vanna_sql] Seeded {len(vn.get_training_data())} training rows", flush=True)
    else:
        print(f"[vanna_sql] Loaded existing training data ({existing} rows)", flush=True)
    return vn


def _seed_training(vn: _YruVanna) -> None:
    """Bootstrap when Chroma is empty. Production seeding uses scripts/seed_vanna_training.py (Phase 2.4)."""
    for table in CORE_TABLES:
        try:
            ddl_df = vn.run_sql(f"SHOW CREATE TABLE `{table}`")
            vn.train(ddl=ddl_df.iloc[0]["Create Table"])
        except Exception as e:
            print(f"[vanna_sql] skip table {table}: {e}", flush=True)
    for doc in BUSINESS_DOCS:
        vn.train(documentation=doc)
    for q, sql in SQL_EXAMPLES:
        vn.train(question=q, sql=sql)


# ── Answer formatting — same instructions as sql_agent, but uses Vanna's own Ollama connection ──
# (Avoids depending on LocalLLMProvider which points to a different model via OPEN_WEBUI_BASE_URL)
_ANSWER_SYSTEM = (
    "คุณคือผู้ช่วยอัจฉริยะที่เชี่ยวชาญข้อมูลภายในของ YRU-AI-RAG "
    "คุณมีหน้าที่ตอบคำถามโดยใช้ข้อมูลที่ให้มาเท่านั้น\n\n"
    "แนวทางการตอบ:\n"
    "- ใช้ภาษาที่เป็นธรรมชาติ เหมือนการสนทนากันระหว่างเพื่อนร่วมงาน\n"
    "- หลีกเลี่ยงการขึ้นต้นประโยคซ้ำๆ เช่น 'จากข้อมูลที่ได้รับ...' แต่ให้เข้าสู่เนื้อหาทันที\n"
    "- สรุปใจความสำคัญให้กระชับ ไม่จำเป็นต้องยกมาทั้งประโยคถ้าไม่จำเป็น\n"
    "- หากข้อมูลไม่เพียงพอ ให้ตอบอย่างสุภาพว่าไม่ทราบข้อมูลนี้ แทนการสร้างข้อมูลขึ้นมาเอง\n"
    "- ถ้ามีหลายรายการให้แสดงเป็น bullet points หรือ Markdown Table\n"
    "- แสดงผลเป็นข้อความธรรมดา หรือ Markdown Table (| col | col |) เท่านั้น ไม่ใช้ JSON หรือ Tag พิเศษ\n"
    "- ปีในข้อมูลทั้งหมดเป็น ปี พ.ศ. (Buddhist Era) ให้แสดงเป็น 'พ.ศ. XXXX' เสมอ ห้ามแปลงเป็น ค.ศ."
)


def _format_answer(query: str, df) -> str:
    if df is None or len(df) == 0:
        return "ไม่พบข้อมูลที่ตรงกับคำถามของคุณในฐานข้อมูลครับ"
    results = df.head(50).to_dict(orient="records")
    vn = _get_vn()
    user_content = (
        f"คำถาม: \"{query}\"\n"
        f"ข้อมูลจากฐานข้อมูล ({len(df)} รายการ):\n{results}\n"
    )
    answer = vn.submit_prompt([
        vn.system_message(_ANSWER_SYSTEM),
        vn.user_message(user_content),
    ])
    # Strip any [SHOW_TABLE...] markers (matches sql_agent behavior)
    return re.sub(r"\[SHOW_TABLE[^\]]*\]", "", answer.strip()).strip()


# ── Public API — same signature as sql_agent.generate_and_run_sql ──
def generate_and_run_sql(query: str) -> str:
    """Drop-in replacement. Same signature and return type as the legacy sql_agent version."""

    # 1. Cache hit (reuses legacy cache — same key across engines for now)
    cached = _get_cached(query)
    if cached:
        return cached

    # 2. Out-of-domain guard (reuse legacy)
    if _is_ood(query):
        return "ขออภัยครับ คำถามนี้อยู่นอกขอบเขตข้อมูลของระบบ YRU-AI-RAG"

    # 3. Normalize BE year
    normalized_query = _normalize_years(query)

    try:
        vn = _get_vn()
    except Exception as e:
        print(f"[vanna_sql] init failed: {type(e).__name__}: {e}", flush=True)
        return "ขออภัยครับ ระบบ Vanna ยังไม่พร้อมใช้งานในขณะนี้"

    # 4. Generate SQL via Vanna (RAG-based few-shot retrieval + LLM)
    try:
        raw_sql = vn.generate_sql(normalized_query, allow_llm_to_see_data=False)
    except Exception as e:
        print(f"[vanna_sql] generate_sql failed: {type(e).__name__}: {e}", flush=True)
        return "ขออภัยครับ ระบบไม่สามารถสร้างคำสั่ง SQL ได้"

    if not raw_sql or not raw_sql.strip():
        return "ขออภัยครับ ระบบไม่สามารถสร้างคำสั่ง SQL ได้"

    # 5. Post-process: fix CE years the LLM may still emit
    sql = _fix_sql_years(raw_sql.strip())

    # 6. Safety: block non-SELECT operations
    if re.search(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", sql, re.IGNORECASE):
        return "ขออภัยครับ คำสั่ง SQL นี้มีความเสี่ยงด้านความปลอดภัย ระบบไม่อนุญาต"

    print(f"[vanna_sql] SQL: {sql[:200]}", flush=True)

    # 7. Execute
    try:
        df = vn.run_sql(sql)
    except Exception as e:
        err = str(e)
        print(f"[vanna_sql] SQL exec failed: {err[:200]}", flush=True)
        return f"ขออภัยครับ เกิดข้อผิดพลาดในการรันคำสั่ง SQL: {err[:150]}"

    # 8. Summarize + cache
    answer = _format_answer(normalized_query, df)
    _set_cache(query, answer)
    return answer


# ── Alias — for callers that prefer explicit "vanna" in the name ──
generate_and_run_sql_vanna = generate_and_run_sql


# ── CLI for local/container testing ──
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    for candidate in (".env.local", ".env"):
        if os.path.exists(candidate):
            load_dotenv(candidate)
            print(f"[cli] loaded {candidate}", flush=True)
            break

    if len(sys.argv) < 2:
        print("Usage: python -m backend.services.vanna_sql \"<คำถาม>\"")
        print("       python -m backend.services.vanna_sql --benchmark  (runs 10 POC questions)")
        sys.exit(1)

    if sys.argv[1] == "--benchmark":
        questions = [
            "โครงการทั้งหมดในปี 2568 มีกี่โครงการ",
            "หน่วยงานไหนได้งบสูงสุดปี 2567",
            "รายชื่อคณะทั้งหมดในระบบ",
            "งบประมาณรวมของคณะครุศาสตร์ปี 2568",
            "โครงการปี 2568 ของคณะวิทยาศาสตร์เทคโนโลยีและการเกษตรมีอะไรบ้าง",
            "โครงการที่ได้งบสูงสุด 5 อันดับปี 2568",
            "ยุทธศาสตร์ทั้งหมดของมหาวิทยาลัยปี 2568",
            "พันธกิจข้อที่ 1 คืออะไร",
            "จำนวนโครงการแต่ละสถานะปี 2567",
            "เปรียบเทียบงบประมาณและจำนวนโครงการปี 2567 กับ 2568",
        ]
        for i, q in enumerate(questions, 1):
            t0 = time.time()
            print(f"\n[{i}/10] {q}")
            try:
                ans = generate_and_run_sql(q)
                print(f"  [{time.time()-t0:.1f}s] {ans[:400]}{'...' if len(ans) > 400 else ''}")
            except Exception as e:
                print(f"  FAIL: {type(e).__name__}: {e}")
    else:
        q = " ".join(sys.argv[1:])
        t0 = time.time()
        answer = generate_and_run_sql(q)
        print(f"\nAnswer ({time.time()-t0:.1f}s):")
        print(answer)
