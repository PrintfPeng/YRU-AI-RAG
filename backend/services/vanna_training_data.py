"""
Vanna training data — schema tables, business rules, few-shot SQL examples.
Kept as pure data module so Phase 2.4 seed script can import + reseed without side effects.
"""
from __future__ import annotations

# ── Curated 12 core tables (matches sql_agent._CORE_TABLES) ──
# ไม่ใส่ทั้ง 97 ตาราง เพราะ LLM 7B ยังสับสน — พิสูจน์จาก sql_agent เดิม
CORE_TABLES: list[str] = [
    "projects",
    "project_template_years",
    "departments",
    "statuses",
    "plans",
    "strategics",
    "missions",
    "outputs",
    "goal_templates",
    "tactic_templates",
    "programs",
    "sdg_templates",
    "project_kpis",
]

# ── Business rules ที่ LLM ต้องรู้ก่อน generate SQL ──
# สกัดจาก sql_agent._TABLE_RELATIONSHIPS + prompt rules + reinforcement จาก POC
BUSINESS_DOCS: list[str] = [
    # Domain overview
    "ระบบนี้คือฐานข้อมูลการวางแผนและงบประมาณของมหาวิทยาลัยราชภัฏยะลา (YRU) — เก็บโครงการ ยุทธศาสตร์ พันธกิจ แผนงาน",

    # Year handling (ค.ศ. vs พ.ศ.)
    "ปีในฐานข้อมูล = ปี พ.ศ. เสมอ (เช่น 2567, 2568) ห้ามแปลงเป็น ค.ศ.",
    "ถ้าผู้ใช้พิมพ์ปีเป็น ค.ศ. (เช่น 2024, 2025) ต้องบวก 543 ก่อนใช้ใน WHERE",
    "ถ้าผู้ใช้ไม่ระบุปี ให้ default เป็น 2568",

    # Critical FK relationships (root cause ที่ POC v1 พลาด 5/10)
    "!!!!สำคัญมาก!!!! ตาราง projects ไม่มีคอลัมน์ year เลย ถ้าเห็น p.year, YEAR(p.year), p.year_id ให้รู้ว่าผิด ปีอยู่ที่ project_template_years.year เท่านั้น",
    "!!!!สำคัญมาก!!!! ถ้าจะกรองปีของโครงการ ต้อง INNER JOIN project_template_years pty ON pty.id = p.project_template_year_id ก่อนเสมอ แล้วใช้ WHERE pty.year = <ปี>",
    "!!!!สำคัญมาก!!!! ตาราง projects ไม่มีคอลัมน์ name ชื่อโครงการอยู่ที่ project_template_years.name — SELECT pty.name AS ชื่อโครงการ ห้ามใช้ p.name",

    # Aggregation patterns
    "การ COUNT โครงการ ห้ามใช้ GROUP BY p.id — ใช้ SELECT COUNT(*) FROM projects p ... WHERE ... โดยไม่มี GROUP BY",
    "budget columns มี 4 คอลัมน์: budget1, budget2, budget3, budget4 (Q1-Q4) ทั้งหมดอยู่ที่ตาราง projects",
    "การหา top-N: SELECT ... ORDER BY <col> DESC LIMIT N — ไม่ต้อง GROUP BY ถ้าไม่ aggregate",

    # status_id gotcha
    "status_id ในตาราง projects เป็น VARCHAR slug ไม่ใช่ integer — JOIN ด้วย: JOIN statuses st ON st.status = p.status_id (ห้ามใช้ st.id = p.status_id)",

    # Department naming rules
    "departments.name เก็บชื่อหน่วยงานภายในเช่น 'คณะครุศาสตร์', 'สำนักงานอธิการบดี', 'คณะวิทยาศาสตร์เทคโนโลยีและการเกษตร'",
    "ห้าม WHERE d.name LIKE '%ราชภัฏยะลา%' หรือ '%YRU%' หรือ '%มหาวิทยาลัย%' เพราะไม่มีหน่วยงานที่ชื่อแบบนี้",
    "ถ้าผู้ใช้ถาม 'โครงการของมหาวิทยาลัย' หรือ 'โครงการของ YRU' = ทุกหน่วยงาน → ไม่ต้อง filter d.name เลย",

    # Semantic entity meaning
    "strategics.name = ชื่อยุทธศาสตร์ (เช่น 'การพัฒนาคุณภาพการศึกษา') — ไม่มีชื่อมหาวิทยาลัยอยู่ในนั้น",
    "missions.name = ชื่อพันธกิจ (เช่น 'ผลิตบัณฑิต', 'วิจัย', 'บริการวิชาการ')",
    "project_kpis มีเฉพาะ: id, project_id, type_id, name, target — ไม่มี department_id, JOIN ต้องผ่าน p.id เท่านั้น",

    # Soft delete
    "ทุก query ต้องมี WHERE ... AND deleted_at IS NULL เพื่อกรอง soft-delete",

    # !!! CRITICAL: table-with-own-year gotcha (Phase 2.6 debug — Q3 fix) !!!
    "!!!!สำคัญมาก!!!! ตาราง strategics, plans, goal_templates, tactic_templates, sdg_templates, project_template_years ทุกตารางนี้ มีคอลัมน์ year ในตัวเองโดยตรง (Direct year column) ห้าม JOIN project_template_years สำหรับตารางเหล่านี้ ใช้ WHERE <table>.year = <ปี> ตรงๆ",
    "การ JOIN project_template_years ใช้เฉพาะกับตาราง projects เท่านั้น (เพราะ projects ไม่มี year column) ตารางอื่นๆ ที่มี year ในตัวแล้ว ห้าม JOIN pty",

    # !!! CRITICAL: departments.type/level meaning (Phase 2.6 debug — Q3 faculty list) !!!
    "!!!!สำคัญมาก!!!! departments.level และ departments.type เป็น int ไม่ใช่ string ห้าม WHERE level='faculty' หรือ type='dept' — ผิด!",
    "departments.type ค่าเป็น int มีความหมาย: type=1 หน่วยงานสนับสนุน (สำนักงานอธิการบดี ฯลฯ), type=2 คณะ (Faculty), type=3 สถาบัน/สำนัก (สำนักวิทยบริการ สถาบันวิจัย), type=9 ศูนย์/หน่วย (ศูนย์ภาษา UBI ฯลฯ)",
    "departments.level ค่าเป็น int: level=1 หน่วยงานหลัก (top-level ไม่มี parent), level=2 ภาควิชา/ฝ่าย (มี parent_id), level=3 หรือลึกกว่านั้นสำหรับ sub-unit",
    "ถ้าถาม 'รายชื่อคณะ' ให้ WHERE type = 2 AND level = 1 (ได้ 5 คณะจริง) ห้ามใช้ LIKE 'คณะ%'",
    "ถ้าถาม 'รายชื่อสำนัก' หมายรวมทั้ง type=1 (สำนักงานอธิการบดี) และ type=3 (สำนักวิทยบริการ) — ใช้ WHERE type IN (1,3) AND level = 1",
    "ถ้าถาม 'ศูนย์' ให้ WHERE type = 9 AND level = 1",
]

# ── Few-shot SQL examples — สกัดจาก sql_agent._SQL_EXAMPLES + POC reinforcement ──
# Format: (natural language question, correct SQL)
SQL_EXAMPLES: list[tuple[str, str]] = [
    # Baseline examples (from sql_agent._SQL_EXAMPLES)
    (
        "รายชื่อหน่วยงานทั้งหมดในระบบ",
        "SELECT id, name AS ชื่อหน่วยงาน, level FROM departments WHERE deleted_at IS NULL ORDER BY level, id LIMIT 50",
    ),
    (
        "นับโครงการแยกตามหน่วยงาน",
        "SELECT d.name AS หน่วยงาน, COUNT(*) AS จำนวนโครงการ "
        "FROM projects p JOIN departments d ON d.id = p.department_id "
        "WHERE p.deleted_at IS NULL GROUP BY d.id, d.name ORDER BY จำนวนโครงการ DESC",
    ),
    (
        "งบประมาณรวมของแต่ละหน่วยงานปี 2566",
        "SELECT d.name AS หน่วยงาน, "
        "SUM(COALESCE(p.budget1,0)+COALESCE(p.budget2,0)+COALESCE(p.budget3,0)+COALESCE(p.budget4,0)) AS งบรวม "
        "FROM projects p "
        "JOIN departments d ON d.id = p.department_id "
        "JOIN project_template_years pty ON pty.id = p.project_template_year_id "
        "WHERE pty.year = 2566 AND p.deleted_at IS NULL "
        "GROUP BY d.id, d.name ORDER BY งบรวม DESC",
    ),
    (
        "ยุทธศาสตร์ทั้งหมดปี 2566",
        "SELECT id, sequence AS ลำดับ, name AS ชื่อยุทธศาสตร์ "
        "FROM strategics WHERE year = 2566 AND deleted_at IS NULL ORDER BY sequence",
    ),
    (
        "จำนวนโครงการปี 2567 แต่ละสถานะ",
        "SELECT st.name AS สถานะ, COUNT(*) AS จำนวนโครงการ "
        "FROM projects p "
        "JOIN statuses st ON st.status = p.status_id "
        "JOIN project_template_years pty ON pty.id = p.project_template_year_id "
        "WHERE pty.year = 2567 AND p.deleted_at IS NULL "
        "GROUP BY st.name ORDER BY จำนวนโครงการ DESC",
    ),

    # Reinforcement examples (จาก POC Q1/Q2/Q5/Q6/Q9 ที่เคยพลาด)
    (
        "โครงการทั้งหมดในปี 2568 มีกี่โครงการ",
        "SELECT COUNT(*) AS จำนวนโครงการ FROM projects p "
        "JOIN project_template_years pty ON pty.id = p.project_template_year_id "
        "WHERE pty.year = 2568 AND p.deleted_at IS NULL",
    ),
    (
        "หน่วยงานไหนได้งบสูงสุดปี 2567",
        "SELECT d.name AS หน่วยงาน, "
        "SUM(COALESCE(p.budget1,0)+COALESCE(p.budget2,0)+COALESCE(p.budget3,0)+COALESCE(p.budget4,0)) AS งบรวม "
        "FROM projects p "
        "JOIN departments d ON d.id = p.department_id "
        "JOIN project_template_years pty ON pty.id = p.project_template_year_id "
        "WHERE pty.year = 2567 AND p.deleted_at IS NULL "
        "GROUP BY d.id, d.name ORDER BY งบรวม DESC LIMIT 1",
    ),
    (
        "โครงการปี 2568 ของคณะครุศาสตร์มีอะไรบ้าง",
        "SELECT pty.name AS ชื่อโครงการ, d.name AS หน่วยงาน "
        "FROM projects p "
        "JOIN departments d ON d.id = p.department_id "
        "JOIN project_template_years pty ON pty.id = p.project_template_year_id "
        "WHERE d.name LIKE '%คณะครุศาสตร์%' AND pty.year = 2568 AND p.deleted_at IS NULL "
        "LIMIT 50",
    ),
    (
        "โครงการที่ได้งบสูงสุด 5 อันดับปี 2568",
        "SELECT pty.name AS ชื่อโครงการ, d.name AS หน่วยงาน, "
        "(COALESCE(p.budget1,0)+COALESCE(p.budget2,0)+COALESCE(p.budget3,0)+COALESCE(p.budget4,0)) AS งบรวม "
        "FROM projects p "
        "JOIN departments d ON d.id = p.department_id "
        "JOIN project_template_years pty ON pty.id = p.project_template_year_id "
        "WHERE pty.year = 2568 AND p.deleted_at IS NULL "
        "ORDER BY งบรวม DESC LIMIT 5",
    ),

    # Phase 2.6 reinforcement — direct-year-column tables (Q3 fix)
    (
        "ยุทธศาสตร์ทั้งหมดของมหาวิทยาลัยปี 2568",
        "SELECT id, sequence AS ลำดับ, name AS ชื่อยุทธศาสตร์ "
        "FROM strategics WHERE year = 2568 AND deleted_at IS NULL ORDER BY sequence",
    ),
    (
        "แผนงานทั้งหมดปี 2568",
        "SELECT id, name AS ชื่อแผนงาน FROM plans WHERE year = 2568 AND deleted_at IS NULL ORDER BY id",
    ),

    # Phase 2.6 reinforcement — departments.type/level meaning
    (
        "รายชื่อคณะทั้งหมดในระบบ",
        "SELECT id, name AS ชื่อคณะ FROM departments "
        "WHERE type = 2 AND level = 1 AND deleted_at IS NULL ORDER BY id",
    ),
    (
        "รายชื่อสำนักทั้งหมด",
        "SELECT id, name AS ชื่อสำนัก, type FROM departments "
        "WHERE type IN (1,3) AND level = 1 AND deleted_at IS NULL ORDER BY type, id",
    ),
    (
        "รายชื่อศูนย์ทั้งหมด",
        "SELECT id, name AS ชื่อศูนย์ FROM departments "
        "WHERE type = 9 AND level = 1 AND deleted_at IS NULL ORDER BY id",
    ),
]
