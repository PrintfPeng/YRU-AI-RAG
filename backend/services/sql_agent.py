# backend/services/sql_agent.py
import os
import re
import time
import hashlib
import mysql.connector
from typing import Optional
from langchain_core.prompts import PromptTemplate
from backend.services.llm_provider import LocalLLMProvider


# ---------------------------------------------------------------------------
# Curated table list: only the 12 core tables the LLM needs to know about.
# Sending all 97 tables confuses the 8B model and causes wrong SQL.
# ---------------------------------------------------------------------------
_CORE_TABLES = [
    'projects',
    'project_template_years',
    'departments',
    'statuses',
    'plans',
    'strategics',
    'missions',
    'outputs',
    'goal_templates',
    'tactic_templates',
    'programs',
    'sdg_templates',
    'project_kpis',   # ตัวชี้วัด KPI ของแต่ละโครงการ (project_id, name, target)
]

# Explicit relationship map injected into the SQL prompt
_TABLE_RELATIONSHIPS = """
=== ความสัมพันธ์ระหว่างตาราง ===
-- Integer FK (JOIN ด้วย id):
projects.project_template_year_id → project_template_years.id   → project_template_years.year (ปี พ.ศ.)
projects.department_id            → departments.id               → departments.name (หน่วยงาน/คณะ/สำนัก)
projects.plan_id                  → plans.id                     → plans.name (แผนงาน)
projects.strategic_id             → strategics.id                → strategics.name (ยุทธศาสตร์)
projects.mission_id               → missions.id                  → missions.name (พันธกิจ)
-- !!! ข้อห้ามสำคัญสำหรับ strategics และ missions !!!
-- strategics.name = ชื่อยุทธศาสตร์ เช่น "ยกระดับคุณภาพมาตรฐานการผลิตบัณฑิต...", "การพัฒนาคุณภาพการศึกษา"
-- ห้ามกรอง: s.name LIKE '%ราชภัฏยะลา%' ← ผิด! ชื่อยุทธศาสตร์ไม่มีชื่อมหาวิทยาลัย
-- ถ้าถาม "ยุทธศาสตร์มหาวิทยาลัย" ให้: SELECT year, name FROM strategics WHERE year=2568 AND deleted_at IS NULL ORDER BY sequence
-- missions.name = ชื่อพันธกิจ เช่น "ผลิตบัณฑิต", "วิจัย", "บริการวิชาการ"
-- ห้ามกรอง: ms.name LIKE '%ราชภัฏยะลา%' ← ผิด!
-- !!! ข้อห้ามสำหรับ departments !!!
-- departments.name = ชื่อหน่วยงาน เช่น 'สำนักงานอธิการบดี', 'คณะวิทยาศาสตร์เทคโนโลยีและการเกษตร'
-- ห้ามกรอง: d.name LIKE '%ยะลา%' หรือ LIKE '%YRU%' หรือ LIKE '%ราชภัฏ%' ← ผิด!
-- เพราะชื่อหน่วยงานไม่มีชื่อมหาวิทยาลัย
-- ถ้าถาม 'งบประมาณ YRU' = งบรวมทั้งมหาวิทยาลัย → ไม่ต้อง filter d.name เลย
projects.output_id                → outputs.id                   → outputs.name (ผลผลิต)       ← กรองชื่อด้วย o.name LIKE '%...%'
projects.tactic_id                → tactic_templates.id          → tactic_templates.name (กลยุทธ์)
projects.goal_id                  → goals.id → goals.goal_template_id → goal_templates.name (เป้าหมาย)
-- *** พิเศษ: status_id เป็น VARCHAR slug ไม่ใช่ integer ***
projects.status_id (varchar)      → statuses.status              → statuses.name (สถานะโครงการ)
-- JOIN ถูกต้อง: JOIN statuses st ON st.status = p.status_id
-- ห้ามใช้:      JOIN statuses st ON st.id = p.status_id  (ผิด!)
-- *** กฎ alias: ถ้า JOIN strategics ใช้ alias "s", ถ้า JOIN statuses ให้ใช้ alias "st" เพื่อไม่ชนกัน ***
-- =========================================================
-- !!! ข้อห้ามสำคัญสำหรับ project_kpis !!!
-- project_kpis มีคอลัมน์แค่: id, project_id, type_id, name, target, requested_by, status_id, note
-- project_kpis ไม่มี department_id !!
-- JOIN project_kpis pk ต้องใช้: ON pk.project_id = p.id เท่านั้น
-- ห้ามเขียน: JOIN departments d ON d.id = pk.department_id (ผิด!)
-- ถูกต้อง: JOIN departments d ON d.id = p.department_id (department_id อยู่ที่ projects ไม่ใช่ project_kpis)
-- =========================================================
=========================================
"""

# SQL examples to guide the LLM toward correct patterns
_SQL_EXAMPLES = """
=== ตัวอย่าง SQL ที่ถูกต้อง ===

-- ดูรายชื่อหน่วยงาน/คณะ/สำนักทั้งหมด:
SELECT id, name AS ชื่อหน่วยงาน, level FROM departments WHERE deleted_at IS NULL ORDER BY level, id;

-- ดูโครงการพร้อมชื่อหน่วยงานและสถานะ (status_id คือ VARCHAR slug):
SELECT p.id, d.name AS หน่วยงาน, s.name AS สถานะ, p.principle, pty.year
FROM projects p
JOIN departments d ON d.id = p.department_id
JOIN statuses s ON s.status = p.status_id
JOIN project_template_years pty ON pty.id = p.project_template_year_id
WHERE pty.year = 2566 LIMIT 10;

-- นับโครงการแยกตามหน่วยงาน:
SELECT d.name AS หน่วยงาน, COUNT(*) AS จำนวนโครงการ
FROM projects p
JOIN departments d ON d.id = p.department_id
WHERE p.deleted_at IS NULL
GROUP BY d.id, d.name ORDER BY จำนวนโครงการ DESC;

-- งบประมาณรวมตามหน่วยงานในปี 2566:
SELECT d.name AS หน่วยงาน,
       SUM(p.budget1 + p.budget2 + p.budget3 + p.budget4) AS งบรวม
FROM projects p
JOIN departments d ON d.id = p.department_id
JOIN project_template_years pty ON pty.id = p.project_template_year_id
WHERE pty.year = 2566 AND p.deleted_at IS NULL
GROUP BY d.id, d.name ORDER BY งบรวม DESC;

-- ดูโครงการในปี 2566 ของหน่วยงานหนึ่ง (กรองด้วย d.name LIKE ไม่ใช่ project_template_years.name):
SELECT p.id, d.name AS หน่วยงาน, st.name AS สถานะ, p.principle
FROM projects p
JOIN departments d ON d.id = p.department_id
JOIN statuses st ON st.status = p.status_id
JOIN project_template_years pty ON pty.id = p.project_template_year_id
WHERE pty.year = 2566 AND d.name LIKE '%สำนักงานอธิการบดี%' AND p.deleted_at IS NULL
LIMIT 20;

-- ดูโครงการในปี 2566 ที่อยู่ในยุทธศาสตร์ที่ชื่อมี "คุณภาพการศึกษา":
SELECT p.id, py.name AS ชื่อโครงการ, d.name AS หน่วยงาน, s.name AS ยุทธศาสตร์
FROM projects p
JOIN strategics s ON s.id = p.strategic_id
JOIN departments d ON d.id = p.department_id
JOIN project_template_years pty ON pty.id = p.project_template_year_id
WHERE pty.year = 2566 AND s.name LIKE '%คุณภาพการศึกษา%' AND p.deleted_at IS NULL
LIMIT 20;

-- ดูโครงการในปี 2566 ที่เกี่ยวกับพันธกิจวิจัย:
SELECT p.id, py.name AS ชื่อโครงการ, d.name AS หน่วยงาน, ms.name AS พันธกิจ
FROM projects p
JOIN missions ms ON ms.id = p.mission_id
JOIN departments d ON d.id = p.department_id
JOIN project_template_years pty ON pty.id = p.project_template_year_id
WHERE pty.year = 2566 AND ms.name LIKE '%วิจัย%' AND p.deleted_at IS NULL
LIMIT 20;

-- ดูโครงการพร้อม hierarchy ครบ (ยุทธศาสตร์, พันธกิจ, แผนงาน, ผลผลิต, เป้าหมาย):
SELECT p.id, py.name AS ชื่อโครงการ, d.name AS หน่วยงาน,
       s.name AS ยุทธศาสตร์, ms.name AS พันธกิจ,
       pl.name AS แผนงาน, o.name AS ผลผลิต
FROM projects p
JOIN departments d ON d.id = p.department_id
JOIN strategics s ON s.id = p.strategic_id
LEFT JOIN missions ms ON ms.id = p.mission_id
LEFT JOIN plans pl ON pl.id = p.plan_id
LEFT JOIN outputs o ON o.id = p.output_id
JOIN project_template_years pty ON pty.id = p.project_template_year_id
WHERE pty.year = 2566 AND p.deleted_at IS NULL
LIMIT 10;

-- *** กฎสำคัญสำหรับการค้นหาชื่อโครงการ ***
-- ชื่อโครงการเก็บใน project_template_years.name
-- ใน DB ไม่มีคำว่า "โครงการ" นำหน้า เช่น เก็บเป็น "พัฒนาการเรียนการสอน" ไม่ใช่ "โครงการพัฒนาการเรียนการสอน"
-- ดังนั้นให้ตัดคำว่า "โครงการ" ออกก่อนแล้วค่อย LIKE search
-- ตัวอย่าง: ค้นหา "โครงการพัฒนาการเรียนการสอน" → ใช้ py.name LIKE '%พัฒนาการเรียนการสอน%'

-- ค้นหาโครงการด้วยชื่อ (strip คำว่า "โครงการ" ออก):
SELECT p.id, py.name AS ชื่อโครงการ, d.name AS หน่วยงาน, py.year AS ปี,
       (COALESCE(p.budget1,0)+COALESCE(p.budget2,0)+COALESCE(p.budget3,0)+COALESCE(p.budget4,0)) AS งบรวม,
       p.principle AS หลักการ, p.objective AS วัตถุประสงค์
FROM projects p
JOIN project_template_years py ON py.id = p.project_template_year_id
JOIN departments d ON d.id = p.department_id
WHERE py.name LIKE '%พัฒนาการเรียนการสอน%' AND p.deleted_at IS NULL
LIMIT 10;

-- ดูยุทธศาสตร์ทั้งหมดในปี 2566:
SELECT id, sequence AS ลำดับ, name AS ชื่อยุทธศาสตร์ FROM strategics
WHERE year = 2566 AND deleted_at IS NULL ORDER BY sequence;

-- ดูพันธกิจทั้งหมด (missions คือพันธกิจหลักของมหาวิทยาลัย ไม่ใช่ชื่อมหาวิทยาลัย):
SELECT id, name AS ชื่อพันธกิจ FROM missions WHERE deleted_at IS NULL ORDER BY id;

-- *** หมายเหตุ: missions.name มีค่าเป็น "ผลิตบัณฑิต", "วิจัย", "บริการวิชาการ" ฯลฯ ***
-- *** ห้ามกรอง missions.name LIKE '%ราชภัฏยะลา%' เพราะจะได้ 0 แถว ***
-- *** ถ้าถามว่า "พันธกิจมีอะไรบ้าง" ให้ SELECT ทั้งหมดโดยไม่มี WHERE เพิ่มเติม ***

-- ดูโครงการที่มีตัวชี้วัด (KPI) เรื่องความพึงพอใจ:
SELECT py.name AS ชื่อโครงการ, pk.name AS ตัวชี้วัด, d.name AS หน่วยงาน
FROM projects p
JOIN project_template_years py ON py.id = p.project_template_year_id
JOIN project_kpis pk ON pk.project_id = p.id
JOIN departments d ON d.id = p.department_id
WHERE pk.name LIKE '%ความพึงพอใจ%'
  AND pk.deleted_at IS NULL
  AND p.deleted_at IS NULL
ORDER BY d.name
LIMIT 20;

-- ดูโครงการที่มีตัวชี้วัด (KPI) ตาม keyword:
SELECT py.name AS ชื่อโครงการ, pk.name AS ตัวชี้วัด
FROM project_kpis pk
JOIN projects p ON p.id = pk.project_id
JOIN project_template_years py ON py.id = p.project_template_year_id
WHERE pk.name LIKE '%[KEYWORD]%' AND pk.deleted_at IS NULL AND p.deleted_at IS NULL
LIMIT 20;
==============================================
"""


# ---------------------------------------------------------------------------
# In-memory Query Result Cache (TTL = 5 minutes)
# ---------------------------------------------------------------------------
_SQL_CACHE: dict = {}
_CACHE_TTL = 300  # seconds


def _cache_key(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def _get_cached(query: str) -> Optional[str]:
    key = _cache_key(query)
    if key in _SQL_CACHE:
        answer, ts = _SQL_CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            print(f"[SQL_Agent] Cache hit: '{query[:60]}'")
            return answer
        del _SQL_CACHE[key]
    return None


def _set_cache(query: str, answer: str):
    if len(_SQL_CACHE) >= 200:
        _SQL_CACHE.pop(next(iter(_SQL_CACHE)))
    _SQL_CACHE[_cache_key(query)] = (answer, time.time())


# ---------------------------------------------------------------------------
# Year Normalization Helpers
# ---------------------------------------------------------------------------

def _normalize_years(query: str) -> str:
    """Convert CE years (<=2200) to BE years by adding 543."""
    def replace_year(m):
        y = int(m.group(0))
        return str(y + 543) if y <= 2200 else str(y)
    return re.sub(r'\b(19|20)\d{2}\b', replace_year, query)


def _fix_sql_years(sql: str) -> str:
    """Post-process generated SQL: fix CE year values (1900-2200) to BE years."""
    def _conv(y: str) -> str:
        year = int(y)
        return str(year + 543) if 1900 <= year <= 2200 else y

    # Fix comparison operators: =, >, <, >=, <=, !=
    sql = re.sub(
        r'([=><!]+\s*)(\d{4})\b',
        lambda m: m.group(1) + _conv(m.group(2)),
        sql
    )
    # Fix BETWEEN xxxx AND xxxx
    sql = re.sub(
        r'\bBETWEEN\s+(\d{4})\s+AND\s+(\d{4})\b',
        lambda m: f"BETWEEN {_conv(m.group(1))} AND {_conv(m.group(2))}",
        sql, flags=re.IGNORECASE
    )
    # Fix IN (xxxx, xxxx, ...)
    def _fix_in(m):
        inner = re.sub(r'\b(\d{4})\b', lambda n: _conv(n.group(1)), m.group(1))
        return f"IN ({inner})"
    sql = re.sub(r'\bIN\s*\(([^)]+)\)', _fix_in, sql, flags=re.IGNORECASE)
    return sql


def _extract_be_year(query: str) -> Optional[int]:
    """Extract BE year from query after normalization."""
    normalized = _normalize_years(query)
    match = re.search(r'\b(25\d\d)\b', normalized)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Fallback SQL Builder (used on retry after LLM generates invalid SQL)
# ---------------------------------------------------------------------------

def _build_fallback_sql(query: str) -> str:
    """Safe fallback SQL — specific checks first, generic be_year last."""
    be_year = _extract_be_year(query)
    q = query.lower()

    # ── 1. Department list ──────────────────────────────────────────────────
    dept_keywords = ['แผนก', 'หน่วยงาน', 'คณะ', 'สำนัก', 'ศูนย์']
    count_keywords = ['กี่', 'จำนวน', 'นับ', 'count', 'ทั้งหมด']
    is_dept  = any(kw in q for kw in dept_keywords)
    is_count = any(kw in q for kw in count_keywords)
    if is_dept and not is_count:
        return (
            "SELECT id, name AS ชื่อหน่วยงาน, level "
            "FROM departments WHERE deleted_at IS NULL ORDER BY level, id LIMIT 30"
        )

    # ── 1b. Count by department — 'นับโครงการตามคณะ' UI button ──────────────
    if is_count and is_dept:
        be = be_year or 2568
        return (
            f"SELECT d.name AS หน่วยงาน, COUNT(p.id) AS จำนวนโครงการ "
            f"FROM projects p "
            f"JOIN departments d ON d.id = p.department_id "
            f"JOIN project_template_years pty ON pty.id = p.project_template_year_id "
            f"WHERE pty.year = {be} AND p.deleted_at IS NULL "
            f"GROUP BY d.id, d.name ORDER BY จำนวนโครงการ DESC"
        )

    # ── 2. Count queries ────────────────────────────────────────────────────
    if is_count and be_year:
        return (
            f"SELECT COUNT(*) AS total FROM projects p "
            f"JOIN project_template_years pty ON pty.id = p.project_template_year_id "
            f"WHERE pty.year = {be_year} AND p.deleted_at IS NULL"
        )
    if is_count:
        return "SELECT COUNT(*) AS total FROM projects WHERE deleted_at IS NULL"

    # ── 3. Budget / งบประมาณ (BEFORE generic be_year) ──────────────────────
    budget_keywords = ['งบประมาณ', 'งบ', 'budget', 'เงิน', 'ค่าใช้จ่าย']
    is_budget = any(kw in q for kw in budget_keywords)
    if is_budget and be_year:
        # Single-row total — model reads directly, no summing needed
        return (
            f"SELECT {be_year} AS ปี, COUNT(*) AS จำนวนโครงการ, "
            f"SUM(COALESCE(p.budget1,0)+COALESCE(p.budget2,0)+COALESCE(p.budget3,0)+COALESCE(p.budget4,0)) AS งบรวม "
            f"FROM projects p "
            f"JOIN project_template_years pty ON pty.id = p.project_template_year_id "
            f"WHERE pty.year = {be_year} AND p.deleted_at IS NULL"
        )
    if is_budget:
        return (
            "SELECT pty.year AS ปี, COUNT(*) AS จำนวนโครงการ, "
            "SUM(COALESCE(p.budget1,0)+COALESCE(p.budget2,0)+COALESCE(p.budget3,0)+COALESCE(p.budget4,0)) AS งบรวม "
            "FROM projects p "
            "JOIN project_template_years pty ON pty.id = p.project_template_year_id "
            "WHERE p.deleted_at IS NULL "
            "GROUP BY pty.year ORDER BY pty.year DESC LIMIT 10"
        )

    # ── 3b. Year comparison — 'เปรียบเทียบปี' UI button ──────────────────────
    if any(kw in q for kw in ['เปรียบเทียบ', 'เทียบปี', 'เทียบ', 'compare']):
        y1 = (be_year - 1) if be_year else 2567
        y2 = be_year or 2568
        return (
            f"SELECT pty.year AS ปี, COUNT(*) AS จำนวนโครงการ, "
            f"SUM(COALESCE(p.budget1,0)+COALESCE(p.budget2,0)+COALESCE(p.budget3,0)+COALESCE(p.budget4,0)) AS งบรวม "
            f"FROM projects p "
            f"JOIN project_template_years pty ON pty.id = p.project_template_year_id "
            f"WHERE pty.year IN ({y1}, {y2}) AND p.deleted_at IS NULL "
            f"GROUP BY pty.year ORDER BY pty.year"
        )

    # ── 4. KPI / ตัวชี้วัด ──────────────────────────────────────────────────
    kpi_keywords = ['kpi', 'ตัวชี้วัด', 'เป้าหมาย', 'ผลลัพธ์']
    is_kpi = any(kw in q for kw in kpi_keywords)
    if is_kpi and be_year:
        return (
            f"SELECT py.name AS ชื่อโครงการ, pk.name AS ตัวชี้วัด, pk.target AS เป้าหมาย, d.name AS หน่วยงาน "
            f"FROM project_kpis pk "
            f"JOIN projects p ON p.id = pk.project_id "
            f"JOIN project_template_years py ON py.id = p.project_template_year_id "
            f"JOIN departments d ON d.id = p.department_id "
            f"WHERE py.year = {be_year} AND pk.deleted_at IS NULL AND p.deleted_at IS NULL LIMIT 30"
        )
    if is_kpi:
        return (
            "SELECT py.name AS ชื่อโครงการ, pk.name AS ตัวชี้วัด, pk.target AS เป้าหมาย, d.name AS หน่วยงาน "
            "FROM project_kpis pk "
            "JOIN projects p ON p.id = pk.project_id "
            "JOIN project_template_years py ON py.id = p.project_template_year_id "
            "JOIN departments d ON d.id = p.department_id "
            "WHERE pk.deleted_at IS NULL AND p.deleted_at IS NULL LIMIT 30"
        )

    # ── 5. ยุทธศาสตร์ / Strategic ────────────────────────────────────────────
    strategic_keywords = ['ยุทธศาสตร์', 'strategic']
    is_strategic = any(kw in q for kw in strategic_keywords)
    if is_strategic and be_year:
        return (
            f"SELECT year, name AS ยุทธศาสตร์, sequence AS ลำดับ "
            f"FROM strategics WHERE year = {be_year} AND deleted_at IS NULL ORDER BY sequence"
        )
    if is_strategic:
        return (
            "SELECT year, name AS ยุทธศาสตร์ FROM strategics "
            "WHERE deleted_at IS NULL AND year != 9999 ORDER BY year DESC, sequence LIMIT 20"
        )

    # ── 6. พันธกิจ / Mission ─────────────────────────────────────────────────
    if any(kw in q for kw in ['พันธกิจ', 'mission']):
        return "SELECT id, name AS พันธกิจ FROM missions WHERE deleted_at IS NULL ORDER BY id"

    # ── 7. แผนงาน / Plan ─────────────────────────────────────────────────────
    if any(kw in q for kw in ['แผนงาน', 'แผน']):
        return (
            "SELECT p.name AS ชื่อแผนงาน, COUNT(pr.id) AS จำนวนโครงการ "
            "FROM plans p LEFT JOIN projects pr ON pr.plan_id = p.id AND pr.deleted_at IS NULL "
            "WHERE p.deleted_at IS NULL GROUP BY p.id, p.name ORDER BY จำนวนโครงการ DESC LIMIT 20"
        )

    # ── 7a. Generic project list — DEF-4: 'โครงการของมหาวิทยาลัย' type ──────
    # Triggered when: query has 'โครงการ', no year, no dept-specific keyword
    # LLM commonly fails here with d.name LIKE '%ราชภัฏยะลา%' (wrong filter)
    if 'โครงการ' in q and not is_budget and not is_kpi and not is_strategic:
        be = be_year or 2568  # default to latest known year with data
        return (
            f"SELECT py.name AS ชื่อโครงการ, d.name AS หน่วยงาน, "
            f"(COALESCE(p.budget1,0)+COALESCE(p.budget2,0)+COALESCE(p.budget3,0)+COALESCE(p.budget4,0)) AS งบประมาณ "
            f"FROM projects p "
            f"JOIN departments d ON d.id = p.department_id "
            f"JOIN project_template_years py ON py.id = p.project_template_year_id "
            f"WHERE py.year = {be} AND p.deleted_at IS NULL "
            f"ORDER BY d.name LIMIT 30"
        )

    # ── 8. Generic year fallback (LAST) ──────────────────────────────────────
    if be_year:
        return (
            f"SELECT d.name AS หน่วยงาน, COUNT(*) AS จำนวนโครงการ, "
            f"SUM(COALESCE(p.budget1,0)+COALESCE(p.budget2,0)+COALESCE(p.budget3,0)+COALESCE(p.budget4,0)) AS งบรวม "
            f"FROM projects p "
            f"JOIN departments d ON d.id = p.department_id "
            f"JOIN project_template_years pty ON pty.id = p.project_template_year_id "
            f"WHERE pty.year = {be_year} AND p.deleted_at IS NULL "
            f"GROUP BY d.id, d.name ORDER BY งบรวม DESC LIMIT 20"
        )

    return None


# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------

def _db_connect():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "10.10.2.154"),
        user=os.getenv("DB_USER", "ai-sandbox-read"),
        password=os.getenv("DB_PASSWORD", "9IKAjm.R7Qzm_OIZ"),
        database=os.getenv("DB_NAME", "ai-sandbox_db"),
        port=int(os.getenv("DB_PORT", 3306)),
        connect_timeout=10,
        charset='utf8mb4',
    )


def get_db_schema() -> str:
    """
    Return curated schema for the 12 most relevant tables only.
    Sending all 97 tables overwhelms the 8B model and causes wrong SQL.
    """
    conn = None
    try:
        conn = _db_connect()
        cursor = conn.cursor()
        schema_text = ""
        for table in _CORE_TABLES:
            try:
                cursor.execute(f"DESCRIBE `{table}`;")
                columns = cursor.fetchall()
                col_details = [f"{col[0]} ({col[1]})" for col in columns]
                schema_text += f"Table: {table}\nColumns: {', '.join(col_details)}\n\n"
            except Exception:
                pass
        return schema_text
    except Exception as e:
        print(f"[SQL_Agent] Error fetching schema: {e}")
        return ""
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def _run_sql(sql: str) -> list:
    """Execute a SELECT query and return results as list of dicts."""
    conn = _db_connect()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

# ── DEF-2: Out-of-Domain Guard ─────────────────────────────────────────────
_IN_DOMAIN_KW = [
    'มหาวิทยาลัย', 'ราชภัฏ', 'yru', 'มรย',
    'งบประมาณ', 'งบ', 'แผน', 'โครงการ', 'ยุทธศาสตร์',
    'พันธกิจ', 'วิสัยทัศน์', 'ตัวชี้วัด', 'kpi', 'เป้าหมาย',
    'คณะ', 'สำนัก', 'กอง', 'แผนก', 'ฝ่าย', 'หน่วยงาน', 'สาขา',
    'นักศึกษา', 'อาจารย์', 'บุคลากร', 'หลักสูตร', 'การศึกษา',
]
_OOD_KW = [
    'ฝน', 'อากาศ', 'พยากรณ์', 'weather', 'rain', 'storm', 'น้ำท่วม',
    'หุ้น', 'ดัชนี', 'ตลาดหลักทรัพย์', 'stock', 'crypto', 'bitcoin',
    'ฟุตบอล', 'บาสเกตบอล', 'นักกีฬา', 'แมนยู', 'บอลโลก',
    'การเมือง', 'นายกรัฐมนตรี', 'พรรคการเมือง', 'เลือกตั้ง', 'รัฐบาล',
    'ละคร', 'ซีรีส์', 'ดารา', 'เพลง', 'หนัง', 'คอนเสิร์ต',
]
_OOD_REPLY = (
    'ขออภัยครับ ผมรองรับเฉพาะข้อมูลที่เกี่ยวข้องกับมหาวิทยาลัยราชภัฏยะลา '
    'เช่น งบประมาณ แผนยุทธศาสตร์ โครงการ และหน่วยงานต่างๆ ครับ '
    'หากมีคำถามเกี่ยวกับมหาวิทยาลัย ยินดีช่วยเหลือเสมอนะครับ'
)

def _is_ood(query: str) -> bool:
    """True = query is out-of-domain and should be declined immediately."""
    q = query.lower()
    if any(kw in q for kw in _IN_DOMAIN_KW):  # in-domain anchor wins
        return False
    return any(kw in q for kw in _OOD_KW)

# ─────────────────────────────────────────────────────────────────────────────

def generate_and_run_sql(query: str) -> str:
    """
    Full pipeline:
      1. Cache check
      2. Year normalization
      3. LLM -> SQL (curated schema + relationship hints)
      4. SQL execution with auto-retry on errors
      5. LLM -> Thai answer
      6. Cache result
    """
    print("[SQL_Agent] เริ่มกระบวนการ Text-to-SQL...")

    # 1. Cache check
    cached = _get_cached(query)
    if cached:
        return cached

    # 1.5 OOD Guard (DEF-2) -- fast keyword filter, zero LLM cost
    if _is_ood(query):
        print(f'[OOD] Declined: {query[:60]!r}', flush=True)
        return _OOD_REPLY

    # 2. Year normalization
    normalized_query = _normalize_years(query)
    if normalized_query != query:
        print(f"[SQL_Agent] Year normalized: '{query}' -> '{normalized_query}'")

    # 2.5 Pre-route: intercept budget-TOTAL queries before LLM (avoids wrong d.name filters)
    # DEF-3 fix: bypass when query asks for per-dept breakdown or comparison
    _budget_kw = ['งบประมาณ', 'งบรวม', 'งบมหา', 'งบ ปี', 'budget']
    _dept_bypass_kw = [
        'หน่วยงาน', 'คณะ', 'สำนัก', 'แผนก', 'กอง', 'ฝ่าย',  # per-dept keywords
        'สูงสุด', 'น้อยสุด', 'ต่ำสุด', 'มากที่สุด', 'น้อยที่สุด',  # superlatives
        'อันดับ', 'rank', 'เปรียบ', 'เทียบ', 'แยก',  # ranking/comparison
    ]
    _be_year_pre     = _extract_be_year(normalized_query)
    _has_budget_kw   = any(kw in normalized_query for kw in _budget_kw)
    _has_dept_bypass = any(kw in normalized_query for kw in _dept_bypass_kw)
    if _be_year_pre and _has_budget_kw and not _has_dept_bypass:
        _direct_sql = (
            f"SELECT {_be_year_pre} AS ปี, COUNT(*) AS จำนวนโครงการ, "
            f"SUM(COALESCE(p.budget1,0)+COALESCE(p.budget2,0)+COALESCE(p.budget3,0)+COALESCE(p.budget4,0)) AS งบรวม "
            f"FROM projects p "
            f"JOIN project_template_years pty ON pty.id = p.project_template_year_id "
            f"WHERE pty.year = {_be_year_pre} AND p.deleted_at IS NULL"
        )
        print(f'[SQL_Agent] Pre-route budget: {_direct_sql[:80]}', flush=True)
        try:
            _pre_results = _run_sql(_direct_sql)
        except Exception as _e:
            print(f'[SQL_Agent] Pre-route error: {_e}', flush=True)
            _pre_results = None
        if _pre_results and not all(all(v is None for v in r.values()) for r in _pre_results):
            print(f'[SQL_Agent] Pre-route returned {len(_pre_results)} rows', flush=True)
            # Go directly to answer generation with these results
            llm = LocalLLMProvider.get_primary_llm(temperature=0.0)
            answer_prompt = PromptTemplate.from_template(
                "คุณคือผู้ช่วยอัจฉริยะที่เชี่ยวชาญข้อมูลภายในของ YRU-AI-RAG คุณมีหน้าที่ตอบคำถามโดยใช้ข้อมูลที่ให้มาเท่านั้น\n"
                "\n"
                "แนวทางการตอบ:\n"
                "- ใช้ภาษาที่เป็นธรรมชาติ เหมือนการสนทนากันระหว่างเพื่อนร่วมงาน\n"
                "- หลีกเลี่ยงการขึ้นต้นประโยคซ้ำๆ เช่น 'จากข้อมูลที่ได้รับ...' แต่ให้เข้าสู่เนื้อหาทันที\n"
                "- สรุปใจความสำคัญให้กระชับ ไม่จำเป็นต้องยกมาทั้งประโยคถ้าไม่จำเป็น\n"
                "- หากข้อมูลไม่เพียงพอ ให้ตอบอย่างสุภาพว่าไม่ทราบข้อมูลนี้ แทนการสร้างข้อมูลขึ้นมาเอง\n"
                "- ถ้ามีหลายรายการให้แสดงเป็น bullet points หรือ Markdown Table\n"
                "- แสดงผลเป็นข้อความธรรมดา หรือ Markdown Table (| col | col |) เท่านั้น ไม่ใช้ JSON หรือ Tag พิเศษ\n"
                "- ปีในข้อมูลทั้งหมดเป็น ปี พ.ศ. (Buddhist Era) ให้แสดงเป็น 'พ.ศ. XXXX' เสมอ ห้ามแปลงเป็น ค.ศ.\n"
                "\n"
                "คำถาม: \"{query}\"\n"
                "ข้อมูลจากฐานข้อมูล ({count} รายการ):\n{results}\n"
            )
            _resp = (answer_prompt | llm).invoke({
                "query": normalized_query,
                "count": len(_pre_results),
                "results": str(_pre_results[:50]),
            })
            _answer = _resp.content.strip()
            _answer = re.sub(r'\[SHOW_TABLE[^\]]*\]', '', _answer).strip()
            _set_cache(query, _answer)
            return _answer

    # 3. Schema (curated, not all 97 tables)
    schema = get_db_schema()
    if not schema:
        return "ขออภัยครับ ไม่สามารถอ่านโครงสร้างฐานข้อมูลได้ในขณะนี้"

    llm = LocalLLMProvider.get_primary_llm(temperature=0.0)

    # 4. Generate SQL
    sql_prompt = PromptTemplate.from_template(
        "คุณคือ Data Analyst ผู้เชี่ยวชาญ MySQL สำหรับระบบบริหารโครงการมหาวิทยาลัย\n\n"
        "โครงสร้างตารางที่เกี่ยวข้อง:\n{schema}\n"
        "{relationships}\n"
        "{examples}\n"
        "=== กฎสำคัญ ===\n"
        "1. ปีโครงการ: JOIN project_template_years pty ON pty.id = p.project_template_year_id, WHERE pty.year = [ปี พ.ศ.]\n"
        "2. ถ้าต้องการชื่อ ต้อง JOIN ตารางที่เกี่ยวข้อง (departments, statuses, plans, strategics ฯลฯ)\n"
        "3. ปีในฐานข้อมูล = ปี พ.ศ. เสมอ (เช่น 2567, 2566, 2563) ห้ามแปลงเป็น ค.ศ.\n"
        "   ถ้าผู้ใช้พิมพ์ปี ค.ศ. (เช่น 2024) ระบบแปลงให้แล้ว ให้ใช้ค่าในคำถามที่ได้รับโดยตรง\n"
        "4. ถ้าถามเกี่ยวกับหน่วยงาน/แผนก ให้ query จาก departments table โดยตรง\n"
        "5. กรองชื่อหน่วยงาน: ใช้ WHERE d.name LIKE '%ชื่อ%' ห้ามใส่ชื่อหน่วยงานใน project_template_years\n"
        "6. เลือก SELECT เฉพาะคอลัมน์ที่มีอยู่จริงตาม Schema ด้านบน\n"
        "7. ค้นหาชื่อโครงการ: ชื่อเก็บใน py.name (project_template_years.name) — ไม่มีคำว่า 'โครงการ' นำหน้า\n"
        "   ตัวอย่าง: ถ้าผู้ใช้พูดถึง 'โครงการพัฒนาการเรียนการสอน' ให้ใช้ py.name LIKE '%พัฒนาการเรียนการสอน%'\n"
        "8. departments = หน่วยงานภายใน YRU (คณะครุศาสตร์, สำนักงานอธิการบดี ฯลฯ)\n"
        "   ห้าม WHERE d.name LIKE '%ราชภัฏยะลา%' หรือ '%มหาวิทยาลัย%' — ไม่มี dept ชื่อนี้ในระบบ\n"
        "   ถ้าผู้ใช้ถาม 'โครงการของมหาวิทยาลัย' = ถามทุก dept → ไม่ต้อง filter d.name\n"
        "9. ถ้าผู้ใช้ไม่ระบุปี → ใช้ปีล่าสุด 2568 เป็น default สำหรับ WHERE pty.year\n"
        "10. [Rule-10-UI] UI Short Query defaults — ถ้าคำถามสั้น/กำกวม ให้ตีความ Intent ดังนี้:\n"
        "    'ค้นหาโครงการ' → SELECT ชื่อโครงการ, หน่วยงาน FROM projects WHERE pty.year=2568 LIMIT 30\n"
        "    'เปรียบเทียบปี' → SELECT pty.year, COUNT(*) AS จำนวนโครงการ, SUM(งบ) AS งบรวม ... WHERE pty.year IN (2567,2568) GROUP BY pty.year\n"
        "    'นับโครงการตามคณะ' → SELECT d.name AS หน่วยงาน, COUNT(p.id) AS จำนวนโครงการ ... WHERE pty.year=2568 GROUP BY d.id ORDER BY จำนวนโครงการ DESC\n"
        "    'งบประมาณรวม' → SELECT SUM(งบ) AS งบรวม, COUNT(*) AS จำนวนโครงการ FROM projects WHERE pty.year=2568\n"
        "================\n\n"
        "คำถาม: \"{query}\"\n\n"
        "เขียนคำสั่ง SELECT ที่ถูกต้องและเรียบง่ายที่สุด\n"
        "ข้อบังคับ: SELECT เท่านั้น ตอบเป็น SQL ล้วนๆ ไม่ต้องอธิบาย"
    )

    print("[SQL_Agent] กำลังสร้าง SQL...")
    sql_response = (sql_prompt | llm).invoke({
        "schema": schema,
        "relationships": _TABLE_RELATIONSHIPS,
        "examples": _SQL_EXAMPLES,
        "query": normalized_query,
    })
    raw_sql = sql_response.content.strip()

    # Clean markdown fences
    code_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', raw_sql, re.DOTALL | re.IGNORECASE)
    if code_match:
        raw_sql = code_match.group(1).strip()
    else:
        raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()

    select_match = re.search(r'(?i)\bSELECT\b[\s\S]*', raw_sql)
    if not select_match:
        return "ขออภัยครับ ระบบไม่สามารถสร้างคำสั่ง SQL ได้"
    raw_sql = select_match.group(0).strip()

    # Fix CE years that LLM may have written
    raw_sql = _fix_sql_years(raw_sql)

    # Safety: allow SELECT only
    if re.search(r'\b(DROP|DELETE|UPDATE|INSERT)\b', raw_sql, re.IGNORECASE):
        return "ขออภัยครับ คำสั่ง SQL นี้มีความเสี่ยงด้านความปลอดภัย ระบบไม่อนุญาต"

    print(f"[SQL_Agent] SQL: {raw_sql}")

    # 5. Execute with auto-retry on column/table errors
    results = None
    last_error = None

    for attempt in range(2):
        try:
            results = _run_sql(raw_sql)
            break
        except mysql.connector.Error as err:
            last_error = err
            print(f"[SQL_Agent] MySQL Error [{err.errno}] attempt {attempt + 1}: {err.msg}")
            if attempt == 0 and err.errno in (1054, 1064, 1146):
                # Try deterministic fallback SQL before giving up
                fallback_sql = _build_fallback_sql(normalized_query)
                if fallback_sql:
                    raw_sql = fallback_sql
                    print(f"[SQL_Agent] Using safe fallback SQL: {raw_sql[:120]}")
                    continue
                print("[SQL_Agent] No fallback available. Aborting.")
                break
            else:
                break
        except Exception as e:
            last_error = e
            print(f"[SQL_Agent] Error: {e}")
            break

    if results is None:
        if hasattr(last_error, 'errno'):
            if last_error.errno == 1146:
                return "ขออภัยครับ ระบบ AI อ้างอิงตารางที่ไม่มีอยู่จริง"
            elif last_error.errno == 1054:
                return "ขออภัยครับ ระบบ AI อ้างอิงชื่อคอลัมน์ที่ไม่ถูกต้อง"
            elif last_error.errno == 1064:
                return "ขออภัยครับ คำสั่ง SQL มีข้อผิดพลาดทางไวยากรณ์"
        return "ขออภัยครับ ระบบประมวลผลข้อมูลล้มเหลว"

    print(f"[SQL_Agent] ได้ผลลัพธ์ {len(results)} รายการ")

    # Treat empty results OR all-NULL values (e.g. SUM with no matching rows) as no-data
    def _all_null(rows):
        return all(all(v is None for v in row.values()) for row in rows) if rows else True

    if not results or _all_null(results):
        # Try safe fallback SQL before giving up
        fallback_sql2 = _build_fallback_sql(normalized_query)
        if fallback_sql2 and fallback_sql2 != raw_sql:
            print(f"[SQL_Agent] Empty/null result → trying fallback SQL", flush=True)
            try:
                results = _run_sql(fallback_sql2)
                if results and not _all_null(results):
                    print(f"[SQL_Agent] Fallback returned {len(results)} rows", flush=True)
                else:
                    return "ไม่พบข้อมูลที่ตรงกับคำถามของคุณในฐานข้อมูลครับ"
            except Exception:
                return "ไม่พบข้อมูลที่ตรงกับคำถามของคุณในฐานข้อมูลครับ"
        else:
            return "ไม่พบข้อมูลที่ตรงกับคำถามของคุณในฐานข้อมูลครับ"

    # 6. Summarize in Thai
    answer_prompt = PromptTemplate.from_template(
        "คุณคือผู้ช่วยอัจฉริยะที่เชี่ยวชาญข้อมูลภายในของ YRU-AI-RAG คุณมีหน้าที่ตอบคำถามโดยใช้ข้อมูลที่ให้มาเท่านั้น\n"
        "\n"
        "แนวทางการตอบ:\n"
        "- ใช้ภาษาที่เป็นธรรมชาติ เหมือนการสนทนากันระหว่างเพื่อนร่วมงาน\n"
        "- หลีกเลี่ยงการขึ้นต้นประโยคซ้ำๆ เช่น 'จากข้อมูลที่ได้รับ...' แต่ให้เข้าสู่เนื้อหาทันที\n"
        "- สรุปใจความสำคัญให้กระชับ ไม่จำเป็นต้องยกมาทั้งประโยคถ้าไม่จำเป็น\n"
        "- หากข้อมูลไม่เพียงพอ ให้ตอบอย่างสุภาพว่าไม่ทราบข้อมูลนี้ แทนการสร้างข้อมูลขึ้นมาเอง\n"
        "- ถ้ามีหลายรายการให้แสดงเป็น bullet points หรือ Markdown Table\n"
        "- แสดงผลเป็นข้อความธรรมดา หรือ Markdown Table (| col | col |) เท่านั้น ไม่ใช้ JSON หรือ Tag พิเศษ\n"
        "- ปีในข้อมูลทั้งหมดเป็น ปี พ.ศ. (Buddhist Era) ให้แสดงเป็น 'พ.ศ. XXXX' เสมอ ห้ามแปลงเป็น ค.ศ.\n"
        "\n"
        "คำถาม: \"{query}\"\n"
        "ข้อมูลจากฐานข้อมูล ({count} รายการ):\n{results}\n"
    )

    print("[SQL_Agent] กำลังสรุปข้อมูล...")
    response = (answer_prompt | llm).invoke({
        "query": normalized_query,  # ใช้ normalized (BE year) ไม่ใช่ original
        "count": len(results),
        "results": str(results[:50]),
    })
    answer = response.content.strip()
    answer = re.sub(r'\[SHOW_TABLE[^\]]*\]', '', answer).strip()

    # 7. Cache and return
    _set_cache(query, answer)
    return answer
