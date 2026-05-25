# backend/services/query_router.py
import re
from typing import Literal, Optional
from langchain_core.prompts import PromptTemplate
from backend.services.llm_provider import LocalLLMProvider


# ---------------------------------------------------------------------------
# Keyword patterns for fast routing — no LLM call needed
# ---------------------------------------------------------------------------

# Patterns that clearly map to SQL (counting, filtering, stats)
_SQL_PATTERNS = [
    # Project counts / filters
    r'กี่\s*(โครงการ|อัน|ชิ้น|รายการ|คน|หน่วยงาน|คณะ|แผนก)',
    r'จำนวน.*(โครงการ|งบ|รายการ|คน|หน่วยงาน)',
    r'นับ\s*(โครงการ|รายการ)',
    r'โครงการ.*(ใน|ของ|ปี)\s*(25\d\d|20\d\d)',
    r'ปี\s*(25\d\d|20\d\d).*(โครงการ|งบ)',
    r'(ทั้งหมด|รวม).*(โครงการ|รายการ)',
    r'รายชื่อ.*โครงการ',
    r'ขอดู.*โครงการ',
    r'มี.*โครงการ.*กี่',
    r'โครงการ.*มีอะไรบ้าง',
    r'สรุป.*โครงการ.*ปี',
    # Budget
    r'งบประมาณ.*(รวม|ทั้งหมด|สูงสุด|มากที่สุด|น้อยที่สุด)',
    r'(สูงสุด|มากที่สุด|น้อยที่สุด|ต่ำสุด).*(งบ|โครงการ)',
    r'งบ.*เท่าไ',
    r'ใคร.*(งบ|เยอะ|มาก)',
    # Departments / Units
    r'(หน่วยงาน|คณะ|สำนัก|แผนก|ศูนย์).*(มี|อะไร|ทั้งหมด|บ้าง|กี่|รายชื่อ|ขอดู)',
    r'(มี|รายชื่อ|ขอดู).*(หน่วยงาน|คณะ|สำนัก|แผนก|ศูนย์)',
    r'(หน่วยงาน|คณะ).*(ที่มี|มีกี่|ใน.*ระบบ)',
    r'รหัส.*โครงการ',
    # Plans / Strategics / Missions
    r'(แผน|ยุทธศาสตร์|กลยุทธ์|พันธกิจ|ผลผลิต).*(มี|ทั้งหมด|บ้าง|รายชื่อ)',
    r'(มี|รายชื่อ).*(แผน|ยุทธศาสตร์)',
    r'พันธกิจ.*(อะไรบ้าง|ทั้งหมด|มีกี่|รายการ)',
    r'สถานะ.*โครงการ',
    r'โครงการ.*สถานะ',
    # KPI / ตัวชี้วัด
    r'ตัวชี้วัด.*(มี|อะไร|บ้าง|โครงการ|ค้นหา)',
    r'โครงการ.*(ตัวชี้วัด|kpi)',
    r'kpi.*(มี|โครงการ|อะไร|บ้าง)',
]

# Patterns that clearly map to RAG (explanations, policy, descriptions)
_RAG_PATTERNS = [
    r'อธิบาย',
    r'หมายความว่า',
    r'เป้าหมาย.*คือ',
    r'วัตถุประสงค์.*คือ',
    r'นโยบาย',
    r'ยุทธศาสตร์.*ที่\s*\d',
    r'ยุทธศาสตร์.*คือ',
    r'แนวทาง.*คือ',
    r'ความหมาย',
    r'แผนพัฒนา.*คือ',
    r'มีเนื้อหา.*ว่า',
    r'ในเอกสาร',
    r'ตามเอกสาร',
]


def _fast_route(query: str) -> Optional[Literal["sql", "rag"]]:
    """
    Keyword-based routing: ตรวจจับ pattern ที่ชัดเจนโดยไม่ต้องเรียก LLM
    คืนค่า None หากยังไม่สามารถตัดสินใจได้
    """
    q = query.strip()
    for pattern in _SQL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return "sql"
    for pattern in _RAG_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return "rag"
    return None  # Inconclusive — fall through to LLM


def route_query(query: str) -> Literal["sql", "rag"]:
    """
    วิเคราะห์คำถามของผู้ใช้เพื่อตัดสินใจว่าจะส่งไปที่ SQL Agent หรือ RAG Pipeline

    Fast path: keyword matching (ไม่เรียก LLM — เร็วกว่า ~3 วินาที)
    Slow path: LLM-based routing สำหรับคำถามที่ไม่ชัดเจน
    """
    # ─── Fast Path ───────────────────────────────────────────────────────────
    fast_result = _fast_route(query)
    if fast_result:
        print(f"[Router] ⚡ Fast route (keyword): [{fast_result.upper()}]")
        return fast_result

    # ─── Slow Path: LLM ──────────────────────────────────────────────────────
    llm = LocalLLMProvider.get_primary_llm(temperature=0.0)

    prompt_template = """คุณคือระบบจำแนกประเภทคำถาม (Query Router) ตอบสั้นที่สุด

ตัวอย่าง "sql":
  - "มีโครงการกี่อัน"
  - "งบประมาณรวมปี 2565 เท่าไหร่"
  - "โครงการในปี 2566 มีอะไรบ้าง"
  - "หน่วยงานไหนได้งบมากที่สุด"

ตัวอย่าง "rag":
  - "ยุทธศาสตร์ที่ 1 คืออะไร"
  - "อธิบายนโยบายมหาวิทยาลัย"
  - "แผนพัฒนามีเป้าหมายอะไร"
  - "เอกสารนี้พูดถึงอะไร"

คำถาม: "{query}"
ตอบ "sql" หรือ "rag" เท่านั้น:"""

    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm

    try:
        print(f"[Router] 🧠 LLM routing: '{query}'")
        response = chain.invoke({"query": query})
        result = response.content.strip().lower()

        if "sql" in result:
            print("[Router] 🔀 LLM decision: [SQL Agent]")
            return "sql"

        print("[Router] 🔀 LLM decision: [Vector/RAG]")
        return "rag"
    except Exception as e:
        print(f"[Router] ⚠️ Routing error (fallback to RAG): {e}")
        return "rag"
