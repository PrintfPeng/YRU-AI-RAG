# scripts/run_mysql_ingestion.py
"""
YRU AI RAG — MySQL Ingestion Pipeline (v2)
==========================================
สิ่งที่ปรับปรุงจาก v1:
  1. กรองเฉพาะ status_id = 'budget_approved' (โครงการที่อนุมัติแล้ว ~1,313 รายการ)
  2. กรอง year != 9999 (ตัดข้อมูลทดสอบออก)
  3. JOIN ครบ hierarchy:
       missions, outputs, goal_templates (ผ่าน goals),
       tactic_templates, sdg_templates (ผ่าน project_template_years)
  4. รวม KPI ต่อท้าย document (project_kpis.name)
  5. Text format ใหม่: อ่านง่าย ครอบคลุมกว่าเดิม
  6. Metadata ละเอียดขึ้น: strategic, mission, status
  7. Clear collection ก่อน re-ingest ทุกครั้ง (ข้อมูลสะอาด)
"""

import sys
import os
import gc
import logging
from pathlib import Path
from decimal import Decimal
from typing import List, Dict, Any

from dotenv import load_dotenv
import mysql.connector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# แก้ปัญหา Encoding บน Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# ─────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────

def get_db_connection() -> mysql.connector.MySQLConnection:
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "10.10.2.154"),
        user=os.getenv("DB_USER", "ai-sandbox-read"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "ai-sandbox_db"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        connect_timeout=30,
    )


# ─────────────────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────────────────

def get_embedding_model():
    base_url = os.getenv("OPEN_WEBUI_BASE_URL")
    api_key  = os.getenv("OPEN_WEBUI_API_KEY", "ollama")

    if base_url:
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info(f"Using remote embedding (bge-m3) via {base_url}")
            return OpenAIEmbeddings(
                model="bge-m3:latest",
                base_url=base_url,
                api_key=api_key,
                check_embedding_ctx_length=False,
            )
        except Exception as e:
            logger.warning(f"Remote embedding unavailable: {e}. Falling back to local.")

    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info("Using local embedding (paraphrase-multilingual-MiniLM-L12-v2)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


# ─────────────────────────────────────────────────────────────
# ChromaDB Client
# ─────────────────────────────────────────────────────────────

def get_chromadb_client(collection_name: str = "yru_planning_data"):
    import chromadb
    from langchain_chroma import Chroma

    embeddings   = get_embedding_model()
    chroma_host  = os.getenv("CHROMA_SERVER_HOST")
    chroma_port  = os.getenv("CHROMA_SERVER_PORT", "8000")

    if chroma_host:
        logger.info(f"Connecting to remote ChromaDB: {chroma_host}:{chroma_port}")
        client       = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
    else:
        persist_dir = PROJECT_ROOT / "chroma_db"
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using local ChromaDB at {persist_dir}")
        client       = chromadb.PersistentClient(path=str(persist_dir))
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    return client, vector_store


# ─────────────────────────────────────────────────────────────
# Fetch KPIs (bulk, keyed by project_id)
# ─────────────────────────────────────────────────────────────

def fetch_kpis(conn) -> Dict[int, List[str]]:
    """ดึง KPI ทั้งหมดของโครงการที่ budget_approved แล้วจัดเป็น dict"""
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT pk.project_id, pk.name AS kpi_name
        FROM project_kpis pk
        JOIN projects p ON p.id = pk.project_id
        WHERE pk.deleted_at IS NULL
          AND pk.name IS NOT NULL AND pk.name != ''
          AND p.status_id = 'budget_approved'
          AND p.deleted_at IS NULL
        ORDER BY pk.project_id, pk.id
    """)
    kpi_map: Dict[int, List[str]] = {}
    for row in cursor.fetchall():
        pid = row["project_id"]
        kpi_map.setdefault(pid, []).append(row["kpi_name"].strip())
    cursor.close()
    logger.info(f"Fetched KPIs for {len(kpi_map)} projects")
    return kpi_map


# ─────────────────────────────────────────────────────────────
# Fetch Projects (main query — full hierarchy JOIN)
# ─────────────────────────────────────────────────────────────

def fetch_projects(conn) -> List[Dict[str, Any]]:
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT
            p.id,
            py.name          AS template_name,
            py.year,
            d.name           AS department_name,
            s.name           AS strategic_name,
            ms.name          AS mission_name,
            pl.name          AS plan_name,
            o.name           AS output_name,
            gt.name          AS goal_name,
            tt.name          AS tactic_name,
            sdg.name         AS sdg_name,
            p.principle,
            p.objective,
            p.expect,
            p.budget1,
            p.budget2,
            p.budget3,
            p.budget4,
            (COALESCE(p.budget1,0) + COALESCE(p.budget2,0)
             + COALESCE(p.budget3,0) + COALESCE(p.budget4,0)) AS total_budget,
            p.status_id
        FROM projects p
        JOIN project_template_years py  ON py.id  = p.project_template_year_id
        LEFT JOIN departments d         ON d.id   = p.department_id
        LEFT JOIN strategics s          ON s.id   = p.strategic_id
        LEFT JOIN missions ms           ON ms.id  = p.mission_id
        LEFT JOIN plans pl              ON pl.id  = p.plan_id
        LEFT JOIN outputs o             ON o.id   = p.output_id
        LEFT JOIN goals g               ON g.id   = p.goal_id
        LEFT JOIN goal_templates gt     ON gt.id  = g.goal_template_id
        LEFT JOIN tactic_templates tt   ON tt.id  = p.tactic_id
        LEFT JOIN sdg_templates sdg     ON sdg.id = py.sdg_id
        WHERE p.deleted_at IS NULL
          AND p.status_id = 'budget_approved'
          AND py.year IS NOT NULL
          AND py.year != 9999
        ORDER BY py.year, p.id
    """)
    rows = cursor.fetchall()
    cursor.close()
    logger.info(f"Fetched {len(rows)} budget_approved projects from MySQL")
    return rows


# ─────────────────────────────────────────────────────────────
# Text Formatter
# ─────────────────────────────────────────────────────────────

def _budget_str(val) -> str:
    try:
        return f"{float(val):,.2f} บาท"
    except Exception:
        return "0.00 บาท"


def format_project_to_text(row: Dict[str, Any], kpis: List[str]) -> str:
    """
    แปลง row โครงการ + KPIs ให้เป็นข้อความอธิบายแบบธรรมชาติ
    เพื่อให้ embedding model เข้าใจบริบทได้ดีที่สุด
    """
    lines = [
        f"ชื่อโครงการ: {row.get('template_name') or 'ไม่ระบุ'}",
        f"รหัสโครงการ: {row.get('id')}",
        f"ปีงบประมาณ พ.ศ.: {row.get('year')}",
        f"หน่วยงานที่รับผิดชอบ: {row.get('department_name') or 'ไม่ระบุ'}",
    ]

    # ─ Hierarchy block ─────────────────────────────
    hierarchy = []
    if row.get("strategic_name"):
        hierarchy.append(f"ยุทธศาสตร์: {row['strategic_name']}")
    if row.get("mission_name"):
        hierarchy.append(f"พันธกิจ: {row['mission_name']}")
    if row.get("plan_name"):
        hierarchy.append(f"แผนงาน: {row['plan_name']}")
    if row.get("output_name"):
        hierarchy.append(f"ผลผลิต: {row['output_name']}")
    if row.get("goal_name"):
        hierarchy.append(f"เป้าหมาย: {row['goal_name']}")
    if row.get("tactic_name"):
        hierarchy.append(f"กลยุทธ์: {row['tactic_name']}")
    if row.get("sdg_name"):
        hierarchy.append(f"SDG: {row['sdg_name']}")

    if hierarchy:
        lines.append("\nความเชื่อมโยงเชิงยุทธศาสตร์:")
        lines.extend([f"  - {h}" for h in hierarchy])

    # ─ Budget block ────────────────────────────────
    b1 = float(row.get("budget1") or 0)
    b2 = float(row.get("budget2") or 0)
    b3 = float(row.get("budget3") or 0)
    b4 = float(row.get("budget4") or 0)
    total = b1 + b2 + b3 + b4
    if total > 0:
        parts = " + ".join(
            _budget_str(b) for b in [b1, b2, b3, b4] if b > 0
        )
        lines.append(f"\nงบประมาณรวม: {_budget_str(total)} ({parts})")
    else:
        lines.append("\nงบประมาณรวม: 0.00 บาท")

    # ─ Content block ────────────────────────────────
    if principle := str(row.get("principle") or "").strip():
        lines.append(f"\nหลักการและเหตุผล: {' '.join(principle.split())}")
    if objective := str(row.get("objective") or "").strip():
        lines.append(f"วัตถุประสงค์: {' '.join(objective.split())}")
    if expect := str(row.get("expect") or "").strip():
        lines.append(f"ผลลัพธ์ที่คาดหวัง: {' '.join(expect.split())}")

    # ─ KPI block ────────────────────────────────────
    if kpis:
        lines.append("\nตัวชี้วัด (KPI):")
        for kpi in kpis[:10]:  # เก็บสูงสุด 10 KPI ต่อโครงการ
            lines.append(f"  - {kpi}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────

def run_ingestion(collection_name: str = "yru_planning_data"):
    logger.info("=" * 60)
    logger.info("YRU AI RAG — MySQL Ingestion Pipeline v2")
    logger.info("=" * 60)

    # 1. Connect DB
    conn = get_db_connection()

    # 2. Fetch data
    try:
        rows    = fetch_projects(conn)
        kpi_map = fetch_kpis(conn)
    finally:
        conn.close()

    if not rows:
        logger.warning("No projects found. Aborting.")
        return

    # 3. Build documents
    logger.info("Building document texts...")
    texts:     List[str]             = []
    metadatas: List[Dict[str, Any]]  = []
    ids:       List[str]             = []

    for row in rows:
        pid  = int(row["id"])
        kpis = kpi_map.get(pid, [])

        text = format_project_to_text(row, kpis)
        texts.append(text)

        metadatas.append({
            "source":     "mysql_planning",
            "table":      "projects",
            "project_id": pid,
            "year":       int(row["year"]) if row["year"] else 0,
            "department": str(row.get("department_name") or "ไม่ระบุ"),
            "strategic":  str(row.get("strategic_name")  or ""),
            "mission":    str(row.get("mission_name")    or ""),
            "plan":       str(row.get("plan_name")       or ""),
            "output":     str(row.get("output_name")     or ""),
            "goal":       str(row.get("goal_name")       or ""),
            "tactic":     str(row.get("tactic_name")     or ""),
            "status":     str(row.get("status_id")       or ""),
            "budget":     float(row.get("total_budget")  or 0),
            "kpi_count":  len(kpis),
        })
        ids.append(f"project_{pid}")

    logger.info(f"Built {len(texts)} documents")

    # 4. Connect ChromaDB + clear old data
    client, vector_store = get_chromadb_client(collection_name)

    # ล้าง collection เดิมก่อน (re-ingest สะอาด)
    try:
        existing = client.get_collection(collection_name)
        old_count = existing.count()
        if old_count > 0:
            logger.info(f"Clearing old collection '{collection_name}' ({old_count:,} docs)...")
            client.delete_collection(collection_name)
            logger.info("Old collection deleted.")
            # สร้าง vector_store ใหม่หลังลบ
            _, vector_store = get_chromadb_client(collection_name)
    except Exception:
        pass  # ยังไม่มี collection ก็ไม่เป็นไร

    # 5. Ingest in batches
    batch_size  = 100
    total_docs  = len(texts)
    total_batches = (total_docs + batch_size - 1) // batch_size

    logger.info(f"Ingesting {total_docs:,} documents in {total_batches} batches...")

    for i in range(0, total_docs, batch_size):
        end        = min(i + batch_size, total_docs)
        batch_num  = i // batch_size + 1

        try:
            vector_store.add_texts(
                texts=texts[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end],
            )
            logger.info(f"  Batch {batch_num}/{total_batches} — {end}/{total_docs} docs ingested")
        except Exception as e:
            logger.error(f"  Batch {batch_num} FAILED: {e}")

        gc.collect()

    # 6. Verify
    try:
        final_col   = client.get_collection(collection_name)
        final_count = final_col.count()
        logger.info("=" * 60)
        logger.info(f"✅ Ingestion complete!")
        logger.info(f"   Collection : {collection_name}")
        logger.info(f"   Documents  : {final_count:,}")
        logger.info(f"   KPI-linked : {sum(1 for m in metadatas if m['kpi_count'] > 0):,} projects")

        # สถิติแยกตามปี
        from collections import Counter
        year_dist = Counter(m["year"] for m in metadatas)
        logger.info("   By year    :")
        for yr, cnt in sorted(year_dist.items()):
            logger.info(f"     พ.ศ. {yr}: {cnt} โครงการ")
        logger.info("=" * 60)
    except Exception as e:
        logger.warning(f"Verification error: {e}")


if __name__ == "__main__":
    run_ingestion()
