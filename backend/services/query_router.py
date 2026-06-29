# backend/services/query_router.py
import re
from typing import Literal, Optional
from langchain_core.prompts import PromptTemplate
from backend.services.llm_provider import LocalLLMProvider


# ---------------------------------------------------------------------------
# Keyword patterns for fast routing — no LLM call needed
# ---------------------------------------------------------------------------


# Patterns that map to project detail lookup (checked BEFORE _SQL_PATTERNS)
# คำที่บ่งบอกว่าเป็น aggregate/list query ไม่ใช่ specific project
_EXCL = r'(?:ทั้งหมด|ทุก(?:โครงการ)?|รวม|ล่าสุด|ปีล่าสุด|กี่|จำนวน|นับ)'

_DETAIL_PATTERNS = [
    # ขอดูรายละเอียด [ชื่อโครงการ] — ต้องไม่ใช่ aggregate
    r'ขอดูรายละเอียด(?!.*(?:ทั้งหมด|ทุก|รวม|ล่าสุด|กี่|จำนวน))',
    r'แสดงรายละเอียด(?!.*(?:ทั้งหมด|ทุก|รวม)).*โครงการ',
    r'ดูรายละเอียด(?!.*(?:ทั้งหมด|ทุก|รวม)).*โครงการ',
    r'รายละเอียด\s+โครงการ(?!.*(?:ทั้งหมด|ทุก|รวม))',
    r'ขอ(?:ดู)?ข้อมูล\s+โครงการ\s+\S',
    r'โครงการ.*รายละเอียด(?!.*(?:ทั้งหมด|ทุก|รวม))',
    r'โครงการ.*(หลักการ|วัตถุประสงค์|ผลลัพธ์|ตัวชี้วัด|KPI|kpi).*',
    # ขอชื่อโครงการ [ชื่อ] — ต้องไม่ใช่ aggregate
    r'ขอชื่อโครงการ(?!.*(?:ทั้งหมด|ทุก|รวม))',
    r'ขอทราบ(?!.*(?:ทั้งหมด|ทุก|รวม)).*โครงการ',
    r'ข้อมูลโครงการ\s+\S',
    r'บอก(?!.*(?:ทั้งหมด|ทุก|รวม)).*โครงการ',
    r'โครงการ\s+\S+.*คือ',
    r'ช่วยหา.*โครงการ\s+\S',
    # เลือกหมายเลขหลัง disambiguation list
    r'^\s*[1-9]\d?[\s\.]',
    r'^\s*[1-9]\d?\s*$',
    r'^(?:เลือก|ข้อ|เลือกข้อ)\s*[1-9]',
]

# Patterns that clearly map to SQL (counting, filtering, stats)
_SQL_PATTERNS = [
    # Project list / aggregate
    r'โครงการ.*ทั้งหมด',
    r'ทั้งหมด.*โครงการ',
    r'รายการโครงการ',
    r'รายชื่อโครงการ',
    r'(?:แสดง|ดู|รายการ).*โครงการ.*(?:ทั้งหมด|ทุก|รวม)',
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
    # DEF-4: 'โครงการของมหาวิทยาลัย/ราชภัฏ' — fast-route to SQL (avoid LLM wrong dept filter)
    r'โครงการ.*ของ.*(มหาวิทยาลัย|ราชภัฏ|มรย)',
    r'(yru|มรย).*โครงการ',
    # UI short-query buttons — fast-route before LLM routing
    r'เปรียบเทียบ.*(ปี|งบ|โครงการ)',
    r'(ปี|งบ).*เปรียบเทียบ',
    r'นับโครงการ.*(ตาม|แยก).*(คณะ|หน่วยงาน|แผนก)',
    r'^ค้นหาโครงการ$',
    r'^งบประมาณรวม$',
    # KPI / ตัวชี้วัด
    r'ตัวชี้วัด.*(มี|อะไร|บ้าง|โครงการ|ค้นหา)',
    r'โครงการ.*(ตัวชี้วัด|kpi)',
    r'kpi.*(มี|โครงการ|อะไร|บ้าง)',
    # DEF-1: ordinal strategic/mission queries — data in SQL, not vector store
    r'ยุทธศาสตร์.*(ข้อ|ลำดับ).*(ที่\s*)?\d',
    r'ยุทธศาสตร์.*(ข้อแรก|ข้อสุดท้าย|อันดับแรก|อันดับสุดท้าย)',
    r'พันธกิจ.*(ข้อ|ลำดับ).*(ที่\s*)?\d',
    r'พันธกิจ.*(ข้อแรก|ข้อสุดท้าย)',
]

# Patterns that clearly map to RAG (explanations, policy, descriptions, project details)
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
    # Project detail retrieval — best answered from ChromaDB rich text
    r'รายละเอียด.*(โครงการ|แผน|กิจกรรม)',
    r'(โครงการ|แผน).*(รายละเอียด|เนื้อหา|เกี่ยวกับ|คืออะไร|คือ)',
    r'ขอ.*(รายละเอียด|ข้อมูล|ทราบ).*(โครงการ|แผน)',
    r'โครงการ.*(วัตถุประสงค์|หลักการ|ผลลัพธ์|ผลที่คาดหวัง)',
    r'หลักการ.*(โครงการ|และเหตุผล)',
]


def _fast_route(query: str) -> Optional[Literal["sql", "rag", "detail"]]:
    """
    Keyword-based routing: ตรวจจับ pattern ที่ชัดเจนโดยไม่ต้องเรียก LLM
    ลำดับ: detail > sql > rag
    คืนค่า None หากยังไม่สามารถตัดสินใจได้
    """
    q = query.strip()
    # ตรวจ detail ก่อนเสมอ (ป้องกัน "ขอดูรายละเอียด" ถูก sql ดักก่อน)
    for pattern in _DETAIL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return "detail"
    for pattern in _SQL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return "sql"
    for pattern in _RAG_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return "rag"
    return None  # Inconclusive — fall through to LLM


def route_query(query: str) -> Literal["sql", "rag", "detail"]:
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

ตัวอย่าง "sql":
  - "ยุทธศาสตร์ข้อที่ 1 ของมหาวิทยาลัยคืออะไร"
  - "พันธกิจข้อที่ 3 คืออะไร"
ตัวอย่าง "rag":
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

        if "detail" in result:
            print("[Router] 🔀 LLM decision: [Detail]")
            return "detail"
        if "sql" in result:
            print("[Router] 🔀 LLM decision: [SQL Agent]")
            return "sql"

        print("[Router] 🔀 LLM decision: [Vector/RAG]")
        return "rag"
    except Exception as e:
        print(f"[Router] ⚠️ Routing error (fallback to RAG): {e}")
        return "rag"
