from __future__ import annotations

"""
semantic_enricher.py (Universal Semantic Intelligence Edition)

โมดูลเพิ่มความหมายและบริบทให้ข้อมูล (Semantic Enrichment Engine):
1) Section Segmentation: แบ่งโครงสร้างเอกสารตามกลุ่มธุรกิจ (Finance, Legal, HR, Technical)
2) Granular Role Tagging: ระบุหน้าที่ของข้อความ (Title, Clause, Step, Warning, Q&A)
3) Schema Normalization: ปรับหัวตารางให้เป็นมาตรฐาน (Canonical Names) เพื่อการประมวลผลอัตโนมัติ
4) Table Intelligence: วิเคราะห์บทบาทของตาราง (Transaction vs Summary)
5) Data Extraction Bridge: เตรียม Payload สำหรับการ Mapping ข้อมูลเข้าสู่ระบบปลายทาง

Engine Strategy:
- AI-Driven (Primary): ใช้ OpenRouter (Qwen) หรือ Google Gemini (Fallback) เพื่อวิเคราะห์บริบทที่ซับซ้อน
- Rule-based (Secondary/Fail-safe): ใช้ Keyword Heuristics ที่ครอบคลุมหลายโดเมนเพื่อความเร็วและกรณี API ขัดข้อง
"""

from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os
import re

load_dotenv()

# [Infrastructure] รองรับ OpenAI SDK สำหรับการเชื่อมต่อ Custom API (OpenRouter)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# [Infrastructure] รองรับ Google Generative AI สำหรับเป็นระบบสำรอง
try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

from .schema import IngestedDocument, TextBlock, TableBlock

# -------------------------------------------------------------------
# Model Configuration (Default: Qwen 2.5 72B Instruct)
# -------------------------------------------------------------------
LLM_MODEL = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")


def _get_llm_client() -> Optional[OpenAI]:
    """คืนค่า Client สำหรับ Primary AI Engine (OpenRouter)"""
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")
    
    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=45)
    except Exception as e:
        print("[semantic_enricher] Cannot init OpenAI Client:", e)
        return None

def _get_google_client():
    """คืนค่า Client สำหรับ Secondary AI Engine (Google Gemini)"""
    if not HAS_GOOGLE: return None
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')
    except: return None


# =============================================================================
# 1) SECTION TAGGING (Document Structure Analysis)
# =============================================================================

# ป้ายกำกับหมวดหมู่เอกสารมาตรฐาน (Universal Taxonomy)
SECTION_LABELS = [
    "header", 
    "summary", 
    "transactions", 
    "legal_section",      # กลุ่มสัญญาและข้อกำหนด
    "instruction_section", # กลุ่มคู่มือและขั้นตอนการทำงาน
    "work_section",        # กลุ่มงานบริหารจัดการ/HR
    "footer", 
    "qna", 
    "other"
]

_QNA_HINTS = ["ถาม:", "คำถาม", "ข้อที่", "จงตอบ", "เลือกคำตอบ", "question"]

def _looks_like_qna(text: str) -> bool:
    """ตรวจสอบรูปแบบข้อความเชิงถาม-ตอบ (Heuristic Detection)"""
    t = text.replace(" ", "")
    return any(h in t for h in _QNA_HINTS)

def _guess_section_rule(block: TextBlock, index: int, total: int) -> str:
    """
    การแบ่ง Section โดยใช้กฎความสัมพันธ์ (Rule-based Fallback):
    - ใช้ตำแหน่งของ Block (Header/Footer)
    - ใช้ Keyword Matching จากหลากหลายประเภทธุรกิจ (Universal Keywords)
    """
    txt = (block.content or "").strip()
    lower = txt.lower()
    extra = block.extra or {}

    is_heading = bool(extra.get("is_heading"))
    page = getattr(block, "page", None) or 0

    # 1. กลุ่มข้อสอบ / Q&A
    if _looks_like_qna(txt):
        return "qna"
    if any(k in lower for k in ["exam", "test", "quiz", "ข้อสอบ", "แบบฝึกหัด"]):
        return "qna"

    # 2. กลุ่มหัวเอกสาร (Header)
    if is_heading and (index < 10 or page <= 2):
        return "header"
    if index == 0 and len(txt) <= 120:
        return "header"

    # 3. กลุ่มบทสรุป (Executive Summary)
    if any(k in lower for k in ["summary", "สรุป", "overview", "executive summary", "บทคัดย่อ", "abstract"]):
        return "summary"

    # 4. กลุ่มรายการเงิน (Financial Transactions)
    if any(k in lower for k in ["รายการเดินบัญชี", "statement", "movement", "invoice", "receipt", "ใบแจ้งหนี้", "ใบเสร็จ", "tax form"]):
        return "transactions"

    # 5. กลุ่มกฎหมายและสัญญา (Legal/Regulatory)
    if any(k in lower for k in ["contract", "agreement", "สัญญา", "ข้อตกลง", "mou", "official", "ประกาศ", "ระเบียบ", "gazette"]):
        return "legal_section"

    # 6. กลุ่มงานสำนักงาน (HR/Admin)
    if any(k in lower for k in ["resume", "cv", "curriculum vitae", "ประวัติ", "minutes", "บันทึกการประชุม", "agenda", "วาระ"]):
        return "work_section"

    # 7. กลุ่มความรู้และคู่มือ (Knowledge Base/Manual)
    if any(k in lower for k in ["manual", "guide", "handbook", "คู่มือ", "instruction", "methodology", "introduction"]):
        return "instruction_section"

    # 8. กลุ่มส่วนท้าย (Footer/Signatures)
    if any(k in lower for k in ["ลงชื่อ", "ผู้มีอำนาจลงนาม", "ขอแสดงความนับถือ", "signature"]):
        return "footer"

    return "other"


def tag_sections(
    doc: IngestedDocument,
    use_llm: bool = False,
) -> IngestedDocument:
    """
    ดำเนินการระบุ Section ให้กับทุก TextBlock ในเอกสาร:
    - ใช้วิธี Batch Processing ผ่าน LLM เพื่อความแม่นยำสูงสุด
    - มีระบบ Plan A (OpenRouter) และ Plan B (Google) เพื่อความเสถียร
    - ผลลัพธ์จะถูกจัดเก็บใน TextBlock.extra["section"]
    """
    client = _get_llm_client() if use_llm else None
    google_client = _get_google_client() if use_llm else None

    if client or google_client:
        # เตรียมตัวอย่างข้อความ (Batching) เพื่อส่งให้ AI วิเคราะห์
        joined = []
        for i, b in enumerate(doc.texts):
            joined.append(f"[{i}] {b.content}")
        prompt_text = "\n".join(joined[:200]) # จำกัดจำนวนเพื่อป้องกัน Context Overflow

        prompt = f"""
You are a document segmenter. Assign ONE section label from:
{SECTION_LABELS}

- header             : document title / main headings
- summary            : executive summary / overview / abstract
- transactions       : financial items / invoice details / statement rows
- legal_section      : contract terms / clauses / regulations
- instruction_section : user manual / guide steps / procedures
- work_section       : resume details / meeting minutes / agenda
- footer             : signatures / closing
- qna                : exam questions / interview Q&A
- other              : anything else

Format: index: label
Text blocks:
{prompt_text}
"""
        mapping: Dict[int, str] = {}
        success = False

        # --- แผน A: OpenRouter (Primary Analysis) ---
        if client:
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful document analyzer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000
                )
                resp_text = response.choices[0].message.content or ""
                success = True
            except Exception as e:
                print(f"[semantic_enricher] Plan A (Section) failed: {e}")
        
        # --- แผน B: Google Gemini (Reliability Fallback) ---
        if not success and google_client:
            try:
                print("[semantic_enricher] Switching to Plan B (Google) for Section Tagging...")
                response = google_client.generate_content(prompt)
                resp_text = response.text
                success = True
            except Exception as e:
                print(f"[semantic_enricher] Plan B (Section) failed: {e}")

        # ประมวลผลผลลัพธ์จาก AI และนำไป Map เข้ากับ Text Blocks
        if success:
            try:
                for line in resp_text.splitlines():
                    line = line.strip()
                    if not line or ":" not in line: continue
                    idx_str, label = line.split(":", 1)
                    idx_str = idx_str.strip().strip("[]")
                    label = label.strip().lower()
                    try: idx = int(idx_str)
                    except: continue
                    if label not in SECTION_LABELS: label = "other"
                    mapping[idx] = label

                total = len(doc.texts)
                for i, b in enumerate(doc.texts):
                    extra = dict(b.extra or {})
                    # ผสานผลลัพธ์จาก AI เข้ากับตรรกะ Rule-based กรณีที่ AI ตกหล่น
                    extra["section"] = mapping.get(i, _guess_section_rule(b, i, total))
                    b.extra = extra
                return doc
            except Exception as e:
                print("[semantic_enricher] Parsing AI response failed:", e)

    # กรณีล้มเหลวทั้งหมด: ใช้ระบบกฎเกณฑ์ (Rule-based Fallback)
    print("[semantic_enricher] Using Rule-based Fallback for Section Tagging")
    total = len(doc.texts)
    for i, b in enumerate(doc.texts):
        extra = dict(b.extra or {})
        extra["section"] = _guess_section_rule(b, i, total)
        b.extra = extra

    return doc


# =============================================================================
# 2) TEXT ROLE CATEGORIZATION (Semantic Role Labeling)
# =============================================================================

# ป้ายกำกับหน้าที่ของข้อความเพื่อการสร้าง RAG Prompt ที่แม่นยำ
TEXT_ROLE_LABELS = [
    "title", 
    "account_info", 
    "transaction_header", 
    "transaction_row", 
    "legal_clause",       # ข้อสัญญา/มาตรา
    "instruction_step",   # ลำดับการทำงาน
    "warning_note",       # คำเตือนเชิงเทคนิค/ความปลอดภัย
    "note", 
    "footer_text", 
    "qna_question", 
    "qna_answer", 
    "other"
]

def _guess_text_role_rule(block: TextBlock) -> str:
    """
    ระบุบทบาทของข้อความด้วยกฎเกณฑ์ (Deterministic Role Guessing):
    - วิเคราะห์จากรูปแบบอักขระ (Regex)
    - วิเคราะห์จากความสัมพันธ์กับ Section ที่สังกัดอยู่
    """
    txt = (block.content or "").strip()
    lower = txt.lower()
    extra = block.extra or {}
    section = extra.get("section")
    is_heading = extra.get("is_heading", False)

    # 1. กลุ่มคำถามและคำตอบ
    if txt.replace(" ", "").startswith("ถาม:") or "คำถาม" in txt or "question" in lower: return "qna_question"
    if txt.replace(" ", "").startswith("ตอบ:") or "เฉลย" in txt or "answer" in lower: return "qna_answer"
    if "?" in txt and len(txt) < 200: return "qna_question"

    # 2. หัวข้อหลัก (Title)
    if section == "header" and (is_heading or len(txt) < 120): return "title"

    # 3. ข้อมูลทางการเงิน
    if any(k in lower for k in ["เลขที่บัญชี", "account no", "bank"]): return "account_info"
    if any(k in lower for k in ["วันที่", "date", "amount", "balance", "คงเหลือ"]): return "transaction_header"
    if section == "transactions" and 10 <= len(txt) <= 200: return "transaction_row"

    # 4. ข้อกำหนดทางกฎหมาย
    if any(k in lower for k in ["ข้อที่", "article", "clause", "section", "มาตรา"]): return "legal_clause"
    if section == "legal_section" and re.match(r"^\d+\.", txt): return "legal_clause"

    # 5. ข้อมูลเชิงเทคนิคและคู่มือ
    if any(k in lower for k in ["คำเตือน", "warning", "caution", "ข้อควรระวัง", "note:", "หมายเหตุ"]): return "warning_note"
    if any(k in lower for k in ["ขั้นตอนที่", "step", "method", "วิธีทำ"]): return "instruction_step"

    # 6. ข้อมูลประกอบและส่วนท้าย
    if any(k in lower for k in ["หมายเหตุ", "remark"]): return "note"
    if any(k in lower for k in ["ลงชื่อ", "signature"]): return "footer_text"

    return "other"


def categorize_text_blocks(
    doc: IngestedDocument,
    use_llm: bool = False,
) -> IngestedDocument:
    """
    จัดกลุ่มหน้าที่ของข้อความ (Role Classification):
    - ช่วยให้ AI ในระบบ RAG เข้าใจความสำคัญของข้อมูลแต่ละส่วน (เช่น รู้ว่าเป็นคำเตือน หรือข้อมูลทั่วไป)
    - ใช้โครงสร้าง Hybrid AI + Rule-based เหมือนกับ Section Tagging
    """
    client = _get_llm_client() if use_llm else None
    google_client = _get_google_client() if use_llm else None

    if client or google_client:
        joined = []
        for i, b in enumerate(doc.texts[:200]):
            section = (b.extra or {}).get("section", "unknown")
            joined.append(f"[{i}] (section={section}) {b.content}")
        prompt_text = "\n".join(joined)

        prompt = f"""
You are a universal document role classifier. Assign ONE role from:
{TEXT_ROLE_LABELS}

- title             : main headings
- account_info      : bank account / party details
- transaction_header: table headers (date, amount)
- transaction_row   : list item row / transaction line
- legal_clause      : contract terms / articles / regulations
- instruction_step  : manual steps / procedures
- warning_note      : warnings / cautions / important notes
- footer_text       : signatures / closing
- qna_question      : exam question
- qna_answer        : answer key
- other             : normal paragraph

Format: index: role
Text blocks:
{prompt_text}
"""
        mapping: Dict[int, str] = {}
        success = False

        # --- การประมวลผลผ่าน AI Tier ---
        if client:
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful document analyzer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000
                )
                resp_text = response.choices[0].message.content or ""
                success = True
            except Exception as e:
                print(f"[semantic_enricher] Plan A (Role) failed: {e}")

        if not success and google_client:
            try:
                print("[semantic_enricher] Switching to Plan B (Google) for Role Tagging...")
                response = google_client.generate_content(prompt)
                resp_text = response.text
                success = True
            except Exception as e:
                print(f"[semantic_enricher] Plan B (Role) failed: {e}")

        if success:
            try:
                for line in resp_text.splitlines():
                    line = line.strip()
                    if not line or ":" not in line: continue
                    idx_str, label = line.split(":", 1)
                    idx_str = idx_str.strip().strip("[]")
                    label = label.strip().lower()
                    try: idx = int(idx_str)
                    except: continue
                    if label not in TEXT_ROLE_LABELS: label = "other"
                    mapping[idx] = label

                for i, b in enumerate(doc.texts):
                    extra = dict(b.extra or {})
                    extra["role"] = mapping.get(i, _guess_text_role_rule(b))
                    b.extra = extra
                return doc
            except Exception as e:
                print("[semantic_enricher] Parsing AI response failed:", e)

    # Fallback Mechanism
    print("[semantic_enricher] Using Rule-based Fallback for Role Tagging")
    for b in doc.texts:
        extra = dict(b.extra or {})
        extra["role"] = _guess_text_role_rule(b)
        b.extra = extra

    return doc


# =============================================================================
# 4) TABLE NORMALIZATION & ROLE ANALYSIS
# =============================================================================

# Mapping หัวตารางจากภาษาพูด/ภาษาไทย เป็นชื่อสากล (Canonical Header Mapping)
HEADER_NORMALIZATION_MAP = {
    # Finance Domain
    "date": "date", "วันที่": "date", "วันเดือนปี": "date",
    "description": "description", "รายการ": "description", "details": "description",
    "withdrawal": "amount_out", "debit": "amount_out", "จ่าย": "amount_out",
    "deposit": "amount_in", "credit": "amount_in", "รับ": "amount_in",
    "balance": "balance", "คงเหลือ": "balance",
    "amount": "amount", "จำนวนเงิน": "amount",
    # Inventory & General Domain
    "qty": "quantity", "จำนวน": "quantity",
    "unit": "unit", "หน่วย": "unit",
    "price": "unit_price", "ราคา": "unit_price"
}

def _normalize_header_name(h: str) -> str:
    """แปลงชื่อหัวตารางให้เป็นรูปแบบมาตรฐาน (Normalization)"""
    h_clean = (h or "").strip().lower()
    if not h_clean: return ""
    for key, canonical in HEADER_NORMALIZATION_MAP.items():
        if key in h_clean: return canonical
    return h_clean

TABLE_ROLE_LABELS = ["transaction_table", "summary_table", "other_table"]

def _guess_table_role(tb: TableBlock) -> str:
    """วิเคราะห์บทบาทของตารางจากความหนาแน่นของข้อมูล (เช่น ตารางรายการ vs ตารางสรุปผล)"""
    header = getattr(tb, "header", []) or []
    header_lower = [str(h).lower() for h in header]
    header_joined = " ".join(header_lower)

    # ตรวจหาคอลัมน์สำคัญที่บ่งบอกว่าเป็นรายการธุรกรรม
    has_date = any("date" in h or "วันที่" in h for h in header_lower)
    has_amount = any(any(x in h for x in ["amount", "ยอด", "debit", "credit", "balance", "price"]) for h in header_lower)

    if has_date and has_amount: return "transaction_table"
    if any(k in header_joined for k in ["summary", "สรุป", "total", "รวม"]): return "summary_table"
    return "other_table"

def normalize_tables(tables: List[TableBlock]) -> List[TableBlock]:
    """
    กระบวนการปรับจูนตาราง (Table Refinement):
    - แก้ปัญหาตารางไม่มี Header โดยการคาดเดาจากแถวแรก (Header Inference)
    - ปรับชื่อหัวตารางให้เป็นมาตรฐาน (Header Normalization)
    - จัดเก็บประวัติการเปลี่ยนแปลงใน Metadata
    """
    for tb in tables:
        header = list(getattr(tb, "header", []) or [])
        rows = list(getattr(tb, "rows", []) or [])

        # [Inference Logic] หากไม่มีหัวตาราง ให้ตรวจสอบว่าแถวแรกเข้าข่ายเป็นหัวตารางหรือไม่
        if not header and rows:
            first = rows[0]
            text_cells = sum(1 for c in first if re.search(r"[A-Za-z\u0E00-\u0E7F]", str(c)))
            if text_cells >= max(1, len(first) // 2):
                header = [str(c) for c in first]
                rows = rows[1:]
                extra = dict(tb.extra or {})
                extra["header_inferred"] = True
                tb.extra = extra

        # ทำการ Normalize และเก็บข้อมูล Role ของตาราง
        normalized_header = [_normalize_header_name(h) for h in header]
        tb.header = normalized_header
        tb.rows = rows

        extra = dict(tb.extra or {})
        extra_norm = dict(extra.get("header_normalization", {}))
        extra_norm.update({"original_header": header, "normalized_header": normalized_header})
        extra["header_normalization"] = extra_norm
        extra["role"] = _guess_table_role(tb)
        tb.extra = extra

    return tables


# =============================================================================
# 5) DATA EXTRACTION & MAPPING (Structured Payload Preparation)
# =============================================================================

def _parse_float_safe(val: Optional[str]) -> Optional[float]:
    """แปลงค่าข้อความเป็นตัวเลขทศนิยมอย่างปลอดภัย (รองรับสัญลักษณ์ทางการเงิน)"""
    if val is None: return None
    s = str(val).strip().replace(",", "").replace("฿", "")
    # รองรับรูปแบบตัวเลขติดลบในทางบัญชี เช่น (100.00)
    if s.startswith("(") and s.endswith(")"): s = "-" + s[1:-1]
    s = s.replace(" ", "")
    try: return float(s)
    except ValueError: return None

def extract_transactions_from_table(tb: TableBlock) -> List[Dict[str, Any]]:
    """
    สกัดข้อมูลธุรกรรมจากตารางให้มาอยู่ในรูปแบบ Dictionary ที่พร้อมใช้งาน (Key-Value):
    - อ้างอิงจาก Canonical Headers ที่ทำ Normalization ไว้แล้ว
    - คำนวณยอดรวมเบื้องต้นกรณีเอกสารสินค้า (Quantity * Unit Price)
    """
    header = getattr(tb, "header", []) or []
    rows = getattr(tb, "rows", []) or []
    name_to_idx = {h: i for i, h in enumerate(header) if h}
    records = []

    for row in rows:
        def col(name: str) -> Optional[str]:
            idx = name_to_idx.get(name)
            if idx is None or idx >= len(row): return None
            return str(row[idx]).strip()

        date = col("date")
        desc = col("description")
        amount_in = col("amount_in")
        amount_out = col("amount_out")
        amount = col("amount")
        balance = col("balance")

        # [Inventory Logic] คำนวณราคารวมอัตโนมัติหากพบข้อมูลจำนวนและราคาต่อหน่วย
        if not amount:
            qty = _parse_float_safe(col("quantity"))
            u_price = _parse_float_safe(col("unit_price"))
            if qty and u_price: amount = str(qty * u_price)

        if not any([date, desc, amount_in, amount_out, amount, balance]): continue

        # จัดเก็บข้อมูลในรูปแบบ Raw (String) และ Cleaned (Float) เพื่อความยืดหยุ่น
        record = {
            "date_raw": date,
            "description": desc,
            "amount_in_raw": amount_in,
            "amount_out_raw": amount_out,
            "amount_raw": amount,
            "balance_raw": balance,
            "amount_in": _parse_float_safe(amount_in),
            "amount_out": _parse_float_safe(amount_out),
            "amount": _parse_float_safe(amount),
            "balance": _parse_float_safe(balance),
        }
        records.append(record)
    return records

def prepare_mapping_payload(doc: IngestedDocument) -> Dict[str, Any]:
    """
    รวบรวมข้อมูลธุรกรรมทั้งหมดจากเอกสารเพื่อสร้าง Payload สำหรับการบูรณาการข้อมูล (Data Integration):
    - ค้นหาและสกัดข้อมูลจากทุกตารางที่มีบทบาทเป็น Transaction Table
    - คืนค่าผลลัพธ์พร้อม Metadata ของเอกสารต้นฉบับ
    """
    all_transactions = []
    for tb in doc.tables:
        txs = extract_transactions_from_table(tb)
        if txs: all_transactions.extend(txs)

    return {
        "doc_id": doc.metadata.doc_id,
        "doc_type": doc.metadata.doc_type,
        "file_name": doc.metadata.file_name,
        "transactions": all_transactions,
    }