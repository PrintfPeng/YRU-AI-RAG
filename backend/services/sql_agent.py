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
projects.strategic_id             → strategics.id                → strategics.name (ยุทธศาสตร์) ← กรองชื่อด้วย s.name LIKE '%...%'
projects.mission_id               → missions.id                  → missions.name (พันธกิจ)     ← กรองชื่อด้วย ms.name LIKE '%...%'
projects.output_id                → outputs.id                   → outputs.name (ผลผลิต)       ← กรองชื่อด้วย o.name LIKE '%...%'
projects.tactic_id                → tactic_templates.id          → tactic_templates.name (กลยุทธ์)
projects.goal_id                  → goals.id → goals.goal_template_id → goal_templates.name (เป้าหมาย)
-- *** พิเศษ: status_id เป็น VARCHAR slug ไม่ใช่ integer ***
projects.status_id (varchar)      → statuses.status              → statuses.name (สถานะโครงการ)
-- JOIN ถูกต้อง: JOIN statuses st ON st.status = p.status_id
-- ห้ามใช้:      JOIN statuses st ON st.id = p.status_id  (ผิด!)
-- *** กฎ alias: ถ้า JOIN strategics ใช้ alias "s", ถ้า JOIN statuses ให้ใช้ alias "st" เพื่อไม่ชนกัน ***
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
    """Post-process generated SQL: fix CE year values to BE years."""
    def _replace(m):
        prefix, year = m.group(1), int(m.group(2))
        return f"{prefix}{year + 543}" if year <= 2200 else m.group(0)
    return re.sub(r'(=\s*)(\d{4})\b', _replace, sql)


def _extract_be_year(query: str) -> Optional[int]:
    """Extract BE year from query after normalization."""
    normalized = _normalize_years(query)
    match = re.search(r'\b(25\d\d)\b', normalized)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Fallback SQL Builder (used on retry after LLM generates invalid SQL)
# ---------------------------------------------------------------------------

def _build_fallback_sql(query: str) -> str:
    """Safe fallback SQL using only verified columns."""
    be_year = _extract_be_year(query)
    count_keywords = ['กี่', 'จำนวน', 'นับ', 'count', 'รวม', 'ทั้งหมด']
    dept_keywords = ['แผนก', 'หน่วยงาน', 'คณะ', 'สำนัก', 'ศูนย์']
    is_count = any(kw in query for kw in count_keywords)
    is_dept = any(kw in query for kw in dept_keywords)

    if is_dept and not is_count:
        return (
            "SELECT id, name AS ชื่อหน่วยงาน, level "
            "FROM departments WHERE deleted_at IS NULL ORDER BY level, id LIMIT 30"
        )
    if is_count and be_year:
        return (
            f"SELECT COUNT(*) AS total FROM projects p "
            f"JOIN project_template_years pty ON pty.id = p.project_template_year_id "
            f"WHERE pty.year = {be_year} AND p.deleted_at IS NULL"
        )
    if is_count:
        return "SELECT COUNT(*) AS total FROM projects WHERE deleted_at IS NULL"
    if be_year:
        return (
            f"SELECT p.id, d.name AS หน่วยงาน, s.name AS สถานะ, p.principle, "
            f"p.budget1, p.budget2, p.budget3, p.budget4 "
            f"FROM projects p "
            f"JOIN departments d ON d.id = p.department_id "
            f"JOIN statuses s ON s.status = p.status_id "
            f"JOIN project_template_years pty ON pty.id = p.project_template_year_id "
            f"WHERE pty.year = {be_year} LIMIT 20"
        )
    return (
        "SELECT p.id, d.name AS หน่วยงาน, s.name AS สถานะ, p.principle "
        "FROM projects p "
        "JOIN departments d ON d.id = p.department_id "
        "JOIN statuses s ON s.status = p.status_id "
        "LIMIT 20"
    )


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

    # 2. Year normalization
    normalized_query = _normalize_years(query)
    if normalized_query != query:
        print(f"[SQL_Agent] Year normalized: '{query}' -> '{normalized_query}'")

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
        "3. ใช้ค่าปีในคำถามโดยตรง ห้ามแปลงค่าปี\n"
        "4. ถ้าถามเกี่ยวกับหน่วยงาน/แผนก ให้ query จาก departments table โดยตรง\n"
        "5. กรองชื่อหน่วยงาน: ใช้ WHERE d.name LIKE '%ชื่อ%' ห้ามใส่ชื่อหน่วยงานใน project_template_years\n"
        "6. เลือก SELECT เฉพาะคอลัมน์ที่มีอยู่จริงตาม Schema ด้านบน\n"
        "7. ค้นหาชื่อโครงการ: ชื่อเก็บใน py.name (project_template_years.name) — ไม่มีคำว่า 'โครงการ' นำหน้า\n"
        "   ตัวอย่าง: ถ้าผู้ใช้พูดถึง 'โครงการพัฒนาการเรียนการสอน' ให้ใช้ py.name LIKE '%พัฒนาการเรียนการสอน%'\n"
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
                raw_sql = _build_fallback_sql(query)
                raw_sql = _fix_sql_years(raw_sql)
                print(f"[SQL_Agent] Retrying with fallback SQL: {raw_sql}")
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

    if not results:
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
        "- ห้ามแสดง raw JSON ห้ามใช้ Tag [SHOW_TABLE:...] ใดๆ\n"
        "\n"
        "คำถาม: \"{query}\"\n"
        "ข้อมูลจากฐานข้อมูล ({count} รายการ):\n{results}\n"
    )

    print("[SQL_Agent] กำลังสรุปข้อมูล...")
    response = (answer_prompt | llm).invoke({
        "query": query,
        "count": len(results),
        "results": str(results[:50]),
    })
    answer = response.content.strip()
    answer = re.sub(r'\[SHOW_TABLE[^\]]*\]', '', answer).strip()

    # 7. Cache and return
    _set_cache(query, answer)
    return answer
