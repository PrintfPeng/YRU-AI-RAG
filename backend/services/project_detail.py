# backend/services/project_detail.py
"""
Project Detail Service
======================
ดึงข้อมูลรายละเอียดโครงการเต็มรูปแบบจาก MySQL
รองรับ:
  - ค้นหาโครงการจากชื่อ (LIKE)
  - แสดง department disambiguation ถ้าพบหลายหน่วยงาน
  - แสดง year disambiguation ถ้า 1 หน่วยงานแต่หลายปี
  - แสดงรายละเอียดเต็มรูปแบบ (เหมือน ChromaDB document)
  - รองรับ department + year filter ในคำถามเดียว
"""

import os
import re
import mysql.connector
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DB Connection
# ─────────────────────────────────────────────────────────────────────────────

def _db_connect():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "rag_mysql"),
        user=os.getenv("DB_USER", "ai-sandbox"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "ai_sandbox_db_local"),
        port=int(os.getenv("DB_PORT", 3306)),
        connect_timeout=15,
        charset="utf8",          # utf8mb3 — prevents collation mismatch with utf8mb3_unicode_ci columns
    )


# ─────────────────────────────────────────────────────────────────────────────
# Query Parsing
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_PREFIXES = [
    r'ขอดูรายละเอียด\s*',
    r'แสดงรายละเอียด\s*',
    r'ดูรายละเอียด\s*',
    r'รายละเอียด\s*',
    r'ขอข้อมูล\s*',
    r'ขอดูข้อมูล\s*',
    r'ข้อมูล(?=.*โครงการ)\s*',
    r'ขอดู\s*',
    r'แสดง\s*',
    r'ขอชื่อ\s*',
    r'ขอทราบ(?:ข้อมูล|รายละเอียด|ชื่อ)?\s*',
    r'บอก(?:ข้อมูล|รายละเอียด|ชื่อ)?\s*',
    r'ช่วยบอก(?:ข้อมูล|รายละเอียด)?\s*',
    r'ช่วยหา\s*',
    r'ค้นหา(?:ข้อมูล|รายละเอียด)?\s*',
]

_DEPT_PATTERNS = [
    r'\s+หน่วยงาน(?:ที่รับผิดชอบ)?(?:\s*(?:เป็น|คือ|:))?\s+(.+?)(?:\s+ปี\s+\d+)?$',
    r'\s+ของ\s+((?:คณะ|สำนัก|ศูนย์|สถาบัน|โรงเรียน|หน่วย).+?)(?:\s+ปี\s+\d+)?$',
    # "...ชื่อโครงการ คณะX" — ไม่มี keyword นำหน้า แต่ชื่อหน่วยงานขึ้นต้นด้วยคำที่รู้จัก
    r'\s+((?:คณะ|สำนัก|ศูนย์|สถาบัน|โรงเรียน|หน่วยงาน|กอง|ฝ่าย)\S+(?:\s+\S+){0,4}?)(?:\s+ปี\s+\d+)?$',
]


def _extract_year(query: str) -> Optional[int]:
    """ดึงปีงบประมาณ (พ.ศ.) จากคำถาม"""
    m = re.search(r'\b(25\d\d)\b', query)
    if m:
        return int(m.group(1))
    m2 = re.search(r'\b(20\d\d)\b', query)
    if m2:
        return int(m2.group(1)) + 543
    return None


def _parse_query(query: str) -> Tuple[str, Optional[str], Optional[int]]:
    """
    แยก project name, department filter, year filter จากคำถาม
    Returns: (project_name, dept_filter, year_filter)
    """
    text = query.strip()

    # ดึง year filter ก่อน (จากคำถามต้นฉบับ)
    year_filter = _extract_year(text)

    # ตัด prefix ออก
    for prefix in _QUERY_PREFIXES:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE).strip()

    # หา department filter
    dept_filter: Optional[str] = None
    for pattern in _DEPT_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            dept_filter = m.group(1).strip()
            text = text[: m.start()].strip()
            break

    # ตัด suffix ที่ไม่ใช่ชื่อโครงการ (คืออะไร, คือ, ?)
    text = re.sub(r'\s*(?:คือ(?:อะไร)?|\?)\s*$', '', text).strip()

    # ตัด "ปี XXXX" ออกจากชื่อโครงการ
    text = re.sub(r'\s+ปี\s+\d+', '', text).strip()

    # ตัด "โครงการ" นำหน้า
    text = re.sub(r'^โครงการ\s*', '', text).strip()

    return text, dept_filter, year_filter


# ─────────────────────────────────────────────────────────────────────────────
# Database Queries
# ─────────────────────────────────────────────────────────────────────────────

def _search_by_name(project_name: str) -> list:
    """ค้นหาโครงการจากชื่อ"""
    conn = _db_connect()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT
                p.id,
                py.name  AS project_name,
                py.year,
                d.name   AS department_name,
                (COALESCE(p.budget1,0) + COALESCE(p.budget2,0)
                 + COALESCE(p.budget3,0) + COALESCE(p.budget4,0)) AS total_budget
            FROM projects p
            JOIN project_template_years py ON py.id = p.project_template_year_id
            JOIN departments d             ON d.id  = p.department_id
            WHERE py.name LIKE %s
              AND p.deleted_at IS NULL
            ORDER BY py.year DESC, d.name
            LIMIT 100
        """, (f"%{project_name}%",))
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


def _fetch_full_detail(project_id: int) -> Tuple[Optional[dict], list]:
    """ดึงรายละเอียดเต็มรูปแบบ + KPIs"""
    conn = _db_connect()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT
                p.id,
                py.name   AS template_name,
                py.year,
                d.name    AS department_name,
                s.name    AS strategic_name,
                ms.name   AS mission_name,
                pl.name   AS plan_name,
                o.name    AS output_name,
                gt.name   AS goal_name,
                tt.name   AS tactic_name,
                sdg.name  AS sdg_name,
                p.principle,
                p.objective,
                p.expect,
                p.budget1, p.budget2, p.budget3, p.budget4,
                (COALESCE(p.budget1,0) + COALESCE(p.budget2,0)
                 + COALESCE(p.budget3,0) + COALESCE(p.budget4,0)) AS total_budget
            FROM projects p
            JOIN project_template_years py ON py.id = p.project_template_year_id
            LEFT JOIN departments d        ON d.id   = p.department_id
            LEFT JOIN strategics s         ON s.id   = p.strategic_id
            LEFT JOIN missions ms          ON ms.id  = p.mission_id
            LEFT JOIN plans pl             ON pl.id  = p.plan_id
            LEFT JOIN outputs o            ON o.id   = p.output_id
            LEFT JOIN goals g              ON g.id   = p.goal_id
            LEFT JOIN goal_templates gt    ON gt.id  = g.goal_template_id
            LEFT JOIN tactic_templates tt  ON tt.id  = p.tactic_id
            LEFT JOIN sdg_templates sdg    ON sdg.id = py.sdg_id
            WHERE p.id = %s AND p.deleted_at IS NULL
        """, (project_id,))
        row = cur.fetchone()

        kpis = []
        if row:
            cur.execute("""
                SELECT name FROM project_kpis
                WHERE project_id = %s
                  AND deleted_at IS NULL
                  AND name IS NOT NULL AND name != ''
                ORDER BY id
            """, (project_id,))
            kpis = [r["name"] for r in cur.fetchall()]

        return row, kpis
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Formatter
# ─────────────────────────────────────────────────────────────────────────────

def _budget_str(val) -> str:
    try:
        return f"{float(val):,.2f} บาท"
    except Exception:
        return "0.00 บาท"


def _format_detail(row: dict, kpis: list) -> str:
    lines = [
        f"ชื่อโครงการ: {row.get('template_name') or 'ไม่ระบุ'}",
        f"รหัสโครงการ: {row.get('id')}",
        f"ปีงบประมาณ พ.ศ.: {row.get('year')}",
        f"หน่วยงานที่รับผิดชอบ: {row.get('department_name') or 'ไม่ระบุ'}",
    ]

    hierarchy = []
    for label, key in [
        ("ยุทธศาสตร์", "strategic_name"),
        ("พันธกิจ",    "mission_name"),
        ("แผนงาน",     "plan_name"),
        ("ผลผลิต",     "output_name"),
        ("เป้าหมาย",   "goal_name"),
        ("กลยุทธ์",    "tactic_name"),
        ("SDG",        "sdg_name"),
    ]:
        if row.get(key):
            hierarchy.append(f"{label}: {row[key]}")

    if hierarchy:
        lines.append("\nความเชื่อมโยงเชิงยุทธศาสตร์:")
        lines.extend(f"  - {h}" for h in hierarchy)

    b1 = float(row.get("budget1") or 0)
    b2 = float(row.get("budget2") or 0)
    b3 = float(row.get("budget3") or 0)
    b4 = float(row.get("budget4") or 0)
    total = b1 + b2 + b3 + b4
    if total > 0:
        parts = " + ".join(_budget_str(b) for b in [b1, b2, b3, b4] if b > 0)
        lines.append(f"\nงบประมาณรวม: {_budget_str(total)} ({parts})")
    else:
        lines.append("\nงบประมาณรวม: 0.00 บาท")

    for label, key in [
        ("หลักการและเหตุผล", "principle"),
        ("วัตถุประสงค์",     "objective"),
        ("ผลลัพธ์ที่คาดหวัง", "expect"),
    ]:
        value = str(row.get(key) or "").strip()
        if value:
            lines.append(f"\n{label}: {' '.join(value.split())}")

    if kpis:
        lines.append("\nตัวชี้วัด (KPI):")
        lines.extend(f"  - {kpi}" for kpi in kpis)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Handler
# ─────────────────────────────────────────────────────────────────────────────


# ── Disambiguation cache (thread-safe via GIL for simple dict ops) ──────────
import threading as _threading
_DISAMBIG_LOCK = _threading.Lock()
_DISAMBIG_CACHE = {}   # key="last" → list of project name strings

def _save_disambig(options: list):
    """Save disambiguation options (max 9) for follow-up chip generation."""
    with _DISAMBIG_LOCK:
        _DISAMBIG_CACHE["last"] = list(options[:99])

def _get_disambig() -> list:
    with _DISAMBIG_LOCK:
        return list(_DISAMBIG_CACHE.get("last", []))

def _clear_disambig():
    with _DISAMBIG_LOCK:
        _DISAMBIG_CACHE.pop("last", None)

def handle_project_detail(query: str) -> str:
    """
    Entry point สำหรับ project detail queries

    Flow:
      1. แยก project name, dept filter, year filter
      2. ค้นหาใน MySQL
      3a. ไม่พบ → แจ้งผู้ใช้
      3b. หลาย dept → department disambiguation
      3c. 1 dept, หลายปี → year disambiguation
      3d. 1 dept, 1 ปี → แสดงรายละเอียดเต็ม
    """
    # ── Number selection: user replied with "1", "2", "1. DeptName?" etc. ──
    import re as _re_num
    _text_stripped = query.strip()
    _nm = _re_num.match(r'^\s*([1-9]\d?)[\s\.、]', _text_stripped) or \
          _re_num.match(r'^\s*([1-9]\d?)\s*$', _text_stripped)
    if _nm:
        _sel_idx = int(_nm.group(1)) - 1
        _cached = _get_disambig()
        if _cached and 0 <= _sel_idx < len(_cached):
            _chosen = _cached[_sel_idx]
            _clear_disambig()
            return handle_project_detail(_chosen)
        elif not _cached:
            return "ไม่มีรายการที่รอการเลือก กรุณาถามชื่อโครงการที่ต้องการก่อนครับ"
        else:
            return f"กรุณาเลือกหมายเลข 1–{len(_cached)} ครับ"

    project_name, dept_filter, year_filter = _parse_query(query)

    # คำ generic ที่บ่งบอกว่าไม่ใช่ชื่อโครงการเฉพาะ
    _GENERIC = {'ทั้งหมด', 'ทุก', 'รวม', 'ล่าสุด', 'ปีล่าสุด', 'ใหม่ล่าสุด',
                'ทุกโครงการ', 'ทุกหน่วยงาน', 'กี่', 'จำนวน'}

    if not project_name or any(t in project_name for t in _GENERIC) or len(project_name) <= 2:
        return (
            "กรุณาระบุ **ชื่อโครงการ** ที่ต้องการดูรายละเอียดครับ\n"
            "ตัวอย่าง: \"ขอดูรายละเอียด โครงการพัฒนาการเรียนการสอน\"\n\n"
            "หากต้องการดูรายการโครงการทั้งหมด ลองถามว่า:\n"
            "\"โครงการปี 2567 ทั้งหมดมีกี่โครงการ\" หรือ \"รายชื่อโครงการปี 2567\""
        )

    try:
        matches = _search_by_name(project_name)
    except Exception as _db_err:
        import traceback
        print(f"[project_detail] DB error: {_db_err}", flush=True)
        return (
            f"เกิดข้อผิดพลาดในการเชื่อมต่อฐานข้อมูลครับ\n"
            f"({type(_db_err).__name__}: {str(_db_err)[:120]})"
        )

    if not matches:
        return (
            f"ไม่พบโครงการที่มีชื่อตรงกับ **\"{project_name}\"** ในฐานข้อมูลครับ\n"
            "ลองตรวจสอบชื่อโครงการหรือค้นหาด้วยคำอื่นครับ"
        )

    # ── กรอง department ─────────────────────────────────────────────────────
    if dept_filter:
        # ลำดับ: exact → startswith → substring
        exact = [m for m in matches
                 if m["department_name"].lower() == dept_filter.lower()]
        if exact:
            matches = exact
        else:
            starts = [m for m in matches
                      if m["department_name"].lower().startswith(dept_filter.lower())]
            if starts:
                matches = starts
            else:
                sub = [m for m in matches
                       if dept_filter.lower() in m["department_name"].lower()]
                if sub:
                    matches = sub

    # ── กรอง year ───────────────────────────────────────────────────────────
    if year_filter:
        year_matches = [m for m in matches if int(m["year"]) == year_filter]
        if year_matches:
            matches = year_matches

    # ── จัดกลุ่มตาม department ──────────────────────────────────────────────
    by_dept: dict = {}
    for m in matches:
        by_dept.setdefault(m["department_name"], []).append(m)

    # ─── กรณี 1 department ──────────────────────────────────────────────────
    if len(by_dept) == 1:
        dept_name    = next(iter(by_dept.keys()))
        dept_matches = next(iter(by_dept.values()))
        dept_matches.sort(key=lambda x: x["year"])

        # 1 ปี → แสดงรายละเอียดทันที
        if len(dept_matches) == 1:
            row, kpis = _fetch_full_detail(dept_matches[0]["id"])
            if not row:
                return "ไม่สามารถดึงข้อมูลรายละเอียดโครงการได้ครับ"
            return _format_detail(row, kpis)

        # หลายปี → year disambiguation
        # บันทึก year-specific queries ลง cache
        _year_queries = [
            f"{project_name} หน่วยงาน {dept_name} ปี {m['year']}"
            for m in dept_matches
        ]
        _save_disambig(_year_queries)

        lines = [
            f"พบโครงการ **\"{project_name}\"** ของ **{dept_name}** "
            f"ใน **{len(dept_matches)} ปีงบประมาณ** กรุณาเลือกปีที่ต้องการ:\n",
        ]
        for i, m in enumerate(dept_matches, 1):
            budget = float(m.get("total_budget") or 0)
            bstr   = f"{budget:,.0f} บาท" if budget > 0 else "ไม่มีงบ"
            lines.append(f"  **{i}. พ.ศ. {m['year']}** — งบประมาณ: {bstr}")
        lines.append(
            f"\n💡 พิมพ์หมายเลข (1, 2, 3...) หรือระบุปีงบประมาณ เช่น:\n"
            f"  \"ขอดูรายละเอียด โครงการ{project_name} ปี {dept_matches[-1]['year']} หน่วยงาน {dept_name}\""
        )
        return "\n".join(lines)

    # ─── กรณีหลาย department → disambiguation ───────────────────────────────
    total_found = sum(len(v) for v in by_dept.values())
    # บันทึก dept-specific queries ลง cache เพื่อ follow-up chips
    _sorted_depts = sorted(by_dept.items())
    _disambig_queries = [f"{project_name} หน่วยงาน {dept}" for dept, _ in _sorted_depts]
    _save_disambig(_disambig_queries)

    lines = [
        f"พบโครงการ **\"{project_name}\"** ใน **{len(by_dept)} หน่วยงาน** "
        f"(รวม {total_found} รายการ) กรุณาเลือกหมายเลขหน่วยงาน:\n",
    ]

    for i, (dept, dept_matches) in enumerate(_sorted_depts, 1):
        years  = sorted({str(m["year"]) for m in dept_matches})
        total_budget = sum(float(m.get("total_budget") or 0) for m in dept_matches)
        year_str     = ", ".join(f"พ.ศ. {y}" for y in years)
        budget_disp  = f"{total_budget:,.0f} บาท" if total_budget > 0 else "ไม่มีงบ"
        lines.append(
            f"**{i}. {dept}**\n"
            f"   ปีงบประมาณ: {year_str}\n"
            f"   งบประมาณรวม: {budget_disp}\n"
        )

    lines.append(
        f'\n💡 กรุณาพิมพ์หมายเลข (1, 2, 3...) หรือระบุหน่วยงาน เช่น:\n'
        f'"ขอดูรายละเอียด โครงการ{project_name} หน่วยงาน [ชื่อหน่วยงาน]"'
    )
    return "\n".join(lines)
