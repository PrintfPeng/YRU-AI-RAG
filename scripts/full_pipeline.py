#!/usr/bin/env python3
"""
YRU AI RAG — Full Data Pipeline
=================================
รัน Data Cleaning ทั้ง 4 Phase + Re-ingest เข้า ChromaDB แบบอัตโนมัติ

  Phase 1 : ลบ Test Data (year = 9999)
  Phase 2 : Archive + ลบ Soft-Deleted Records (deleted_at IS NOT NULL)
  Phase 3 : Fix Orphan FK (id = 0 → NULL)
  Phase 4 : Text Normalization (trim, empty→NULL)
  Ingest  : Re-ingest → ChromaDB (yru_planning_data)

Usage (from host):
  docker exec hybrid_rag_backend python3 /app/scripts/full_pipeline.py
  docker exec hybrid_rag_backend python3 /app/scripts/full_pipeline.py --dry-run
  docker exec hybrid_rag_backend python3 /app/scripts/full_pipeline.py --skip-ingest
  docker exec hybrid_rag_backend python3 /app/scripts/full_pipeline.py --only-ingest
  docker exec hybrid_rag_backend python3 /app/scripts/full_pipeline.py --phase 3

หมายเหตุ:
  - รัน script นี้ซ้ำได้ (idempotent) — ถ้าข้อมูลสะอาดแล้วก็จะไม่มีการเปลี่ยนแปลง
  - --dry-run จะแสดงว่า "จะทำอะไร" แต่ไม่แก้ข้อมูลจริง
  - ลำดับการทำงาน: Phase 1 → 2 → 3 → 4 → Ingest (ลำดับสำคัญมาก)
"""

import sys
import os
import re
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# แก้ปัญหา Encoding บน Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

import mysql.connector

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    line = "=" * 65
    logger.info(line)
    logger.info(f"  {title}")
    logger.info(line)


def get_connection() -> mysql.connector.MySQLConnection:
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "rag_mysql"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER", "ai-sandbox"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "ai_sandbox_db_local"),
        charset="utf8mb4",
        connect_timeout=30,
    )


def column_exists(cur, table: str, column: str) -> bool:
    cur.execute("""
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME   = %s
          AND COLUMN_NAME  = %s
    """, (table, column))
    return cur.fetchone()[0] > 0


def table_exists(cur, table: str) -> bool:
    cur.execute("""
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME   = %s
    """, (table,))
    return cur.fetchone()[0] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — ลบ Test Data (year = 9999)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_remove_test_data(conn, dry_run: bool = False) -> int:
    banner("Phase 1 — ลบ Test Data (year = 9999)")
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) FROM projects p
        JOIN project_template_years py ON py.id = p.project_template_year_id
        WHERE py.year = 9999
    """)
    count = cur.fetchone()[0]
    logger.info(f"พบ Test Data (year=9999): {count:,} rows")

    if count == 0:
        logger.info("ไม่มี Test Data → ข้ามขั้นตอนนี้ ✅")
        cur.close()
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] จะลบ {count:,} rows")
        cur.close()
        return count

    cur.execute("""
        DELETE FROM projects
        WHERE project_template_year_id IN (
            SELECT id FROM project_template_years WHERE year = 9999
        )
    """)
    deleted = cur.rowcount
    conn.commit()
    logger.info(f"✅ ลบ Test Data เสร็จ: {deleted:,} rows")
    cur.close()
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Archive + ลบ Soft-Deleted Records
# ─────────────────────────────────────────────────────────────────────────────

# FK columns ที่ต้องทำให้ nullable ใน archive (เพื่อ idempotency หลัง Phase 3)
_NULLABLE_FK_COLS = [
    "tactic_id", "goal_id", "strategic_id",
    "plan_id", "output_id", "mission_id",
    "depv2_id", "staff_id", "transfer_allocate_id",
]


def phase2_archive_soft_deleted(conn, dry_run: bool = False) -> int:
    banner("Phase 2 — Archive + ลบ Soft-Deleted Records")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM projects WHERE deleted_at IS NOT NULL")
    count = cur.fetchone()[0]
    logger.info(f"พบ Soft-Deleted records: {count:,} rows")

    if count == 0:
        logger.info("ไม่มี Soft-Deleted → ข้ามขั้นตอนนี้ ✅")
        cur.close()
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] จะ archive {count:,} rows → projects_archived แล้วลบออกจาก projects")
        cur.close()
        return count

    # สร้าง archive table (copy structure จาก projects)
    if not table_exists(cur, "projects_archived"):
        cur.execute("CREATE TABLE projects_archived LIKE projects")
        conn.commit()
        logger.info("สร้างตาราง projects_archived ใหม่")
    else:
        logger.info("ตาราง projects_archived มีอยู่แล้ว")

    # ทำให้ optional FK columns เป็น nullable ใน archive
    # (จำเป็นสำหรับ idempotency: รองรับกรณีรัน script ซ้ำหลัง Phase 3 แก้ id=0→NULL แล้ว)
    for col in _NULLABLE_FK_COLS:
        if column_exists(cur, "projects_archived", col):
            try:
                cur.execute(
                    f"ALTER TABLE projects_archived "
                    f"MODIFY COLUMN `{col}` INT UNSIGNED NULL DEFAULT NULL"
                )
                conn.commit()
            except Exception:
                pass  # อาจ nullable อยู่แล้ว

    # Copy soft-deleted rows ไปยัง archive (INSERT IGNORE เพื่อ skip ถ้ามีอยู่แล้ว)
    cur.execute("""
        INSERT IGNORE INTO projects_archived
        SELECT * FROM projects WHERE deleted_at IS NOT NULL
    """)
    archived = cur.rowcount
    conn.commit()
    logger.info(f"  Archive: {archived:,} rows → projects_archived")

    # ลบออกจาก main table
    cur.execute("DELETE FROM projects WHERE deleted_at IS NOT NULL")
    deleted = cur.rowcount
    conn.commit()
    logger.info(f"✅ ลบ Soft-Deleted เสร็จ: {deleted:,} rows จาก projects")

    cur.close()
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Fix Orphan FK (id = 0 → NULL)
# ─────────────────────────────────────────────────────────────────────────────

# FK columns ที่ต้องตรวจ (ยกเว้น required columns เช่น project_template_year_id, department_id)
_FK_CHECK_COLS = [
    "strategic_id",
    "tactic_id",
    "goal_id",
    "mission_id",
    "plan_id",
    "output_id",
    "depv2_id",
    "staff_id",
    "transfer_allocate_id",
]


def phase3_fix_orphan_fk(conn, dry_run: bool = False) -> int:
    banner("Phase 3 — Fix Orphan FK (id = 0 → NULL)")
    cur = conn.cursor()

    total_fixed = 0

    for col in _FK_CHECK_COLS:
        if not column_exists(cur, "projects", col):
            continue

        # นับ rows ที่มีค่า = 0
        cur.execute(f"SELECT COUNT(*) FROM projects WHERE `{col}` = 0")
        count = cur.fetchone()[0]

        if count == 0:
            logger.info(f"  {col}: ✅ ไม่มี id=0")
            continue

        logger.info(f"  {col} = 0: {count:,} rows → แก้เป็น NULL")

        if dry_run:
            total_fixed += count
            continue

        # ตรวจสอบ nullable — ถ้ายังเป็น NOT NULL ให้ ALTER ก่อน
        cur.execute("""
            SELECT IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME   = 'projects'
              AND COLUMN_NAME  = %s
        """, (col,))
        row = cur.fetchone()
        if row and row[0] == "NO":
            logger.info(f"  ALTER TABLE: `{col}` → NULLABLE")
            cur.execute(
                f"ALTER TABLE projects "
                f"MODIFY COLUMN `{col}` INT UNSIGNED NULL DEFAULT NULL"
            )
            conn.commit()

        # แก้ค่า 0 → NULL
        cur.execute(f"UPDATE projects SET `{col}` = NULL WHERE `{col}` = 0")
        fixed = cur.rowcount
        conn.commit()
        logger.info(f"  ✅ {col}: แก้ {fixed:,} rows → NULL")
        total_fixed += fixed

    if total_fixed > 0:
        logger.info(f"✅ Phase 3 แก้ทั้งหมด: {total_fixed:,} rows")
    else:
        logger.info("✅ Phase 3 ข้อมูลสะอาดอยู่แล้ว (0 changes)")

    cur.close()
    return total_fixed


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Text Normalization
# ─────────────────────────────────────────────────────────────────────────────

_TEXT_COLS = ["principle", "objective", "expect"]


def phase4_text_normalization(conn, dry_run: bool = False) -> int:
    banner("Phase 4 — Text Normalization (trim, empty→NULL)")
    cur = conn.cursor()

    total_changed = 0

    for col in _TEXT_COLS:
        if not column_exists(cur, "projects", col):
            continue

        # นับ rows ที่ต้อง trim
        cur.execute(f"""
            SELECT COUNT(*) FROM projects
            WHERE `{col}` IS NOT NULL
              AND TRIM(`{col}`) != `{col}`
        """)
        trim_count = cur.fetchone()[0]

        # นับ empty string
        cur.execute(f"""
            SELECT COUNT(*) FROM projects
            WHERE `{col}` IS NOT NULL
              AND TRIM(`{col}`) = ''
        """)
        empty_count = cur.fetchone()[0]

        logger.info(f"  {col}: ต้อง trim={trim_count:,}, empty→NULL={empty_count:,}")

        if dry_run:
            total_changed += trim_count + empty_count
            continue

        if trim_count > 0:
            cur.execute(f"""
                UPDATE projects
                SET `{col}` = TRIM(`{col}`)
                WHERE `{col}` IS NOT NULL AND TRIM(`{col}`) != `{col}`
            """)
            conn.commit()

        if empty_count > 0:
            cur.execute(f"""
                UPDATE projects
                SET `{col}` = NULL
                WHERE `{col}` IS NOT NULL AND TRIM(`{col}`) = ''
            """)
            conn.commit()

        total_changed += trim_count + empty_count

    if total_changed > 0:
        logger.info(f"✅ Phase 4 แก้ทั้งหมด: {total_changed:,} rows")
    else:
        logger.info("✅ Phase 4 ข้อมูลสะอาดอยู่แล้ว (0 changes)")

    cur.close()
    return total_changed


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_db_summary(conn) -> None:
    banner("สรุปสถานะ Database หลัง Cleaning")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM projects WHERE deleted_at IS NULL")
    active = cur.fetchone()[0]

    try:
        cur.execute("SELECT COUNT(*) FROM projects_archived")
        archived = cur.fetchone()[0]
    except Exception:
        archived = 0

    cur.execute("""
        SELECT COUNT(*) FROM projects
        WHERE deleted_at IS NULL AND status_id = 'budget_approved'
    """)
    budget_approved = cur.fetchone()[0]

    cur.execute("""
        SELECT py.year, COUNT(*) AS cnt
        FROM projects p
        JOIN project_template_years py ON py.id = p.project_template_year_id
        WHERE p.deleted_at IS NULL
        GROUP BY py.year
        ORDER BY py.year
    """)
    by_year = cur.fetchall()

    logger.info(f"  Active projects : {active:,}")
    logger.info(f"  Archived        : {archived:,}")
    logger.info(f"  Budget Approved : {budget_approved:,}  (จะ ingest เข้า ChromaDB)")
    logger.info("  By year:")
    for yr, cnt in by_year:
        if yr and yr != 9999:
            logger.info(f"    พ.ศ. {yr}: {cnt:,} โครงการ")

    cur.close()


# ─────────────────────────────────────────────────────────────────────────────
# Re-ingest → ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion() -> bool:
    banner("Re-ingest → ChromaDB")

    ingestion_script = PROJECT_ROOT / "scripts" / "run_mysql_ingestion.py"
    if not ingestion_script.exists():
        logger.error(f"ไม่พบ ingestion script: {ingestion_script}")
        return False

    logger.info(f"รัน: {ingestion_script}")
    result = subprocess.run(
        [sys.executable, str(ingestion_script)],
        env=os.environ.copy(),
    )

    if result.returncode == 0:
        logger.info("✅ Re-ingest สำเร็จ")
        return True
    else:
        logger.error(f"❌ Re-ingest ล้มเหลว (exit code {result.returncode})")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="YRU AI RAG — Full Data Pipeline (Cleaning + Re-ingest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ตัวอย่าง:
  รัน pipeline ทั้งหมด:
    docker exec hybrid_rag_backend python3 /app/scripts/full_pipeline.py

  ทดลองดูก่อน (ไม่แก้จริง):
    docker exec hybrid_rag_backend python3 /app/scripts/full_pipeline.py --dry-run

  ทำ cleaning เท่านั้น (ไม่ ingest):
    docker exec hybrid_rag_backend python3 /app/scripts/full_pipeline.py --skip-ingest

  ทำ ingest เท่านั้น:
    docker exec hybrid_rag_backend python3 /app/scripts/full_pipeline.py --only-ingest

  ทำ Phase เดียว (1/2/3/4):
    docker exec hybrid_rag_backend python3 /app/scripts/full_pipeline.py --phase 3
        """,
    )
    parser.add_argument("--dry-run",      action="store_true",
                        help="แสดงผลเท่านั้น ไม่แก้ข้อมูลจริง")
    parser.add_argument("--skip-ingest",  action="store_true",
                        help="ข้ามขั้นตอน Re-ingest ChromaDB")
    parser.add_argument("--only-ingest",  action="store_true",
                        help="ทำแค่ Re-ingest ChromaDB (ข้าม cleaning)")
    parser.add_argument("--phase",        type=int, choices=[1, 2, 3, 4],
                        help="รันเฉพาะ Phase ที่ระบุ (แล้ว skip ingest)")
    args = parser.parse_args()

    start_time = datetime.now()

    banner("YRU AI RAG — Full Data Pipeline")
    logger.info(f"  เริ่มต้น : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  DB Host  : {os.getenv('DB_HOST', 'rag_mysql')}:{os.getenv('DB_PORT', '3306')}")
    logger.info(f"  DB Name  : {os.getenv('DB_NAME', 'ai_sandbox_db_local')}")
    if args.dry_run:
        logger.info("  MODE     : *** DRY RUN — ไม่แก้ข้อมูลจริง ***")

    # ─── Only Ingest ──────────────────────────────────────────────────────────
    if args.only_ingest:
        success = run_ingestion()
        sys.exit(0 if success else 1)

    # ─── Connect DB ───────────────────────────────────────────────────────────
    try:
        conn = get_connection()
        logger.info("  เชื่อมต่อ MySQL สำเร็จ ✅")
    except Exception as e:
        logger.error(f"❌ เชื่อมต่อ MySQL ล้มเหลว: {e}")
        sys.exit(1)

    results = {1: 0, 2: 0, 3: 0, 4: 0}
    phase_fn = {
        1: phase1_remove_test_data,
        2: phase2_archive_soft_deleted,
        3: phase3_fix_orphan_fk,
        4: phase4_text_normalization,
    }

    try:
        phases_to_run = [args.phase] if args.phase else [1, 2, 3, 4]

        for p in phases_to_run:
            results[p] = phase_fn[p](conn, args.dry_run)

        if not args.dry_run:
            print_db_summary(conn)

    except Exception as e:
        logger.error(f"❌ Pipeline ล้มเหลวที่ Phase: {e}")
        conn.close()
        sys.exit(1)
    finally:
        conn.close()

    # ─── Re-ingest ────────────────────────────────────────────────────────────
    ingest_ok = True
    if not args.dry_run and not args.skip_ingest and not args.phase:
        ingest_ok = run_ingestion()

    # ─── Final Summary ────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    banner("สรุปผลการทำงาน")
    logger.info(f"  Phase 1  ลบ Test Data (year=9999)  : {results[1]:>6,} rows")
    logger.info(f"  Phase 2  Archive Soft-Deleted       : {results[2]:>6,} rows")
    logger.info(f"  Phase 3  Fix Orphan FK (0→NULL)     : {results[3]:>6,} rows")
    logger.info(f"  Phase 4  Text Normalization         : {results[4]:>6,} rows")
    if not args.dry_run and not args.skip_ingest and not args.phase:
        status = "✅ สำเร็จ" if ingest_ok else "❌ ล้มเหลว"
        logger.info(f"  Re-ingest ChromaDB                : {status}")
    logger.info(f"  ใช้เวลาทั้งหมด                    : {elapsed:.1f} วินาที")
    if args.dry_run:
        logger.info("  *** DRY RUN — ไม่มีการแก้ข้อมูลจริง ***")
    else:
        logger.info("  ✅ Pipeline เสร็จสมบูรณ์")

    sys.exit(0 if ingest_ok else 1)


if __name__ == "__main__":
    main()
