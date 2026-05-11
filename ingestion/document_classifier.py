from __future__ import annotations

"""
document_classifier.py (Final Universal Edition)

โมดูลวิเคราะห์และจำแนกประเภทเอกสาร (Document Classification Engine):
- ใช้เทคนิค Hybrid Fallback System เพื่อประกันความแม่นยำและความเสถียร (Reliability)
- กระบวนการทำงานแบ่งเป็น 3 ระดับ:
    1) OpenRouter (High-Performance LLM): ใช้โมเดลระดับ SOTA ในการวิเคราะห์บริบทเชิงลึก
    2) Google Gemini (Low-Latency Fallback): ระบบสำรองกรณี Primary API ขัดข้อง หรือต้องการความเร็ว
    3) Rule-based (Deterministic Logic): ระบบกันตาย (Fail-safe) โดยใช้ Keyword Matching ทั้งภาษาไทยและอังกฤษ

หมวดหมู่เอกสารที่รองรับ (Updated Categories):
- Finance: เอกสารการเงิน, ใบแจ้งหนี้, ใบเสร็จ, งบการเงิน
- Legal/Admin: สัญญาทางกฎหมาย, หนังสือราชการ, เอกสารยืนยันตัวตน
- Work/HR: ข้อมูลบุคลากร, บันทึกการประชุม, แผนงาน
- Knowledge: คู่มือเทคนิค, งานวิจัย, สื่อการเรียนการสอน
- General: จดหมายทั่วไป หรือเอกสารที่ไม่เข้าพวก
"""

from typing import List, Optional
from dotenv import load_dotenv
import os
import re

load_dotenv()

from ingestion.schema import IngestedDocument, TextBlock, DocumentMetadata

# -------------------------------------------------------------------
# Client Imports (Dynamic Library Loading)
# -------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

# -------------------------------------------------------------------
# Document Label Set (Universal Taxonomy)
# -------------------------------------------------------------------
CANDIDATE_TYPES = [
    # Finance (กลุ่มเอกสารการเงินและภาษี)
    "invoice",              # ใบแจ้งหนี้ / ใบกำกับภาษี
    "receipt",              # ใบเสร็จรับเงิน
    "financial_statement",  # งบการเงิน / รายงานทางบัญชี / Bank Statement
    "tax_form",             # เอกสารที่เกี่ยวกับภาษี (เช่น ภ.ง.ด.)
    
    # Legal & Official (กลุ่มเอกสารกฎหมายและงานสารบรรณ)
    "contract",             # สัญญา / ข้อตกลง / MOU
    "government_doc",       # หนังสือราชการ / ประกาศ / ระเบียบ
    "id_card",              # เอกสารระบุตัวตน (ID Card / Passport)
    
    # Work & HR (กลุ่มเอกสารบริหารจัดการและทรัพยากรบุคคล)
    "resume",               # ประวัติย่อ / CV
    "meeting_minutes",      # รายงานการประชุม / วาระการประชุม
    "project_plan",         # แผนการดำเนินงาน / TOR
    
    # Knowledge & Tech (กลุ่มเอกสารวิชาการและเทคนิค)
    "manual",               # คู่มือการใช้งาน / Technical Specification
    "research_paper",       # งานวิจัย / บทความทางวิชาการ
    "educational",          # สื่อการสอน / ข้อสอบ / แบบฝึกหัด
    
    # General (กลุ่มเอกสารทั่วไป)
    "correspondence",       # จดหมาย / อีเมล / บันทึกภายใน
    "generic",              # เอกสารทั่วไปที่ไม่สามารถระบุประเภทได้ชัดเจน
]

# -------------------------------------------------------------------
# Model Configuration
# -------------------------------------------------------------------
# กำหนดโมเดลหลักที่จะใช้ผ่าน OpenRouter (Default: Qwen 2.5 72B)
PRIMARY_MODEL = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")

# -------------------------------------------------------------------
# HELPER: API Client Lifecycle Managers
# -------------------------------------------------------------------

def _get_openai_client() -> Optional[OpenAI]:
    """สร้างและตั้งค่า Client สำหรับการเชื่อมต่อ OpenRouter (OpenAI Compatible)"""
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")
    if not api_key: return None
    # ตรวจสอบว่า Library ถูกติดตั้งสมบูรณ์หรือไม่
    if OpenAI is None: return None
    try:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=30)
    except: return None

def _get_google_client():
    """สร้างและตั้งค่า Client สำหรับ Google Gemini API เพื่อใช้เป็นระบบสำรอง"""
    if not HAS_GOOGLE: return None
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        # ใช้โมเดลตระกูล Flash เพื่อประสิทธิภาพด้านความเร็วและประหยัด Cost
        return genai.GenerativeModel('gemini-2.5-flash')
    except: return None

# -------------------------------------------------------------------
# HELPER: Text Sampling (Context Window Optimization)
# -------------------------------------------------------------------

def _collect_sample_text(texts: List[TextBlock], max_chars: int = 4000) -> str:
    """
    สกัดตัวอย่างข้อความจากส่วนต้นของเอกสารเพื่อใช้ในการวิเคราะห์
    - จำกัดจำนวนตัวอักษรเพื่อประหยัด Token และรักษาความเร็ว
    - เน้นเฉพาะส่วน Header และเนื้อหาเริ่มต้นซึ่งมักระบุประเภทเอกสารชัดเจนที่สุด
    """
    chunks = []
    current_len = 0
    # วิเคราะห์จาก 30 บล็อกแรก ซึ่งเพียงพอสำหรับการระบุเจตนาของเอกสารส่วนใหญ่
    for t in texts[:30]: 
        s = (t.content or "").strip()
        if not s: continue
        chunks.append(s)
        current_len += len(s)
        if current_len >= max_chars: break
    return "\n".join(chunks)[:max_chars]

# -------------------------------------------------------------------
# LOGIC: Rule-based Classification (Hard-coded Logic)
# -------------------------------------------------------------------

def classify_document_rule_based(doc: IngestedDocument) -> str:
    """
    การจำแนกประเภทด้วยเงื่อนไข (Deterministic) โดยอาศัย Keyword ในชื่อไฟล์และเนื้อหา
    - ทำหน้าที่เป็น Fail-safe สุดท้ายหากระบบ AI ไม่สามารถใช้งานได้
    - รองรับการค้นหาแบบ Case-insensitive ทั้งไทยและอังกฤษ
    """
    text = _collect_sample_text(doc.texts).lower()
    fname = (doc.metadata.file_name or "").lower()
    combined = f"{fname} {text}"

    # 1. Finance (ตรวจสอบเอกสารการเงิน)
    if any(k in combined for k in ["invoice", "ใบแจ้งหนี้", "tax invoice"]): return "invoice"
    if any(k in combined for k in ["receipt", "ใบเสร็จ", "bill"]): return "receipt"
    if any(k in combined for k in ["statement", "รายการเดินบัญชี", "งบดุล", "balance sheet"]): return "financial_statement"
    if any(k in combined for k in ["tax", "ภาษี", "ภ.ง.ด", "withholding"]): return "tax_form"

    # 2. Legal (ตรวจสอบเอกสารกฎหมายและราชการ)
    if any(k in combined for k in ["contract", "agreement", "สัญญา", "ข้อตกลง", "mou"]): return "contract"
    if any(k in combined for k in ["identification", "passport", "บัตรประชาชน", "citizen id"]): return "id_card"
    if any(k in combined for k in ["official", "ประกาศ", "ระเบียบ", "คำสั่ง", "gazette"]): return "government_doc"

    # 3. Work (ตรวจสอบเอกสารสำนักงาน)
    if any(k in combined for k in ["resume", "cv", "curriculum vitae", "ประวัติย่อ", "experience"]): return "resume"
    if any(k in combined for k in ["minutes", "บันทึกการประชุม", "agenda", "วาระ"]): return "meeting_minutes"
    
    # 4. Knowledge (ตรวจสอบเอกสารวิชาการและคู่มือ)
    if any(k in combined for k in ["manual", "guide", "handbook", "คู่มือ", "instruction", "spec"]): return "manual"
    if any(k in combined for k in ["abstract", "introduction", "methodology", "บทคัดย่อ", "วิจัย"]): return "research_paper"
    if any(k in combined for k in ["exam", "test", "quiz", "ข้อสอบ", "แบบฝึกหัด", "lesson"]): return "educational"

    return "generic"

# -------------------------------------------------------------------
# LOGIC: Hybrid LLM Classification (Advanced Reasoning)
# -------------------------------------------------------------------

def classify_document_with_llm(doc: IngestedDocument) -> str:
    """
    การจำแนกประเภทเชิงลึกด้วย LLM ผ่านระบบ Hybrid:
    - ลำดับความสำคัญ: OpenRouter -> Google Gemini -> Rule-based
    - ใช้ Zero-shot Prompting เพื่อบังคับเอาท์พุตให้ตรงตาม Schema ที่กำหนด
    """
    sample_text = _collect_sample_text(doc.texts)
    # Edge Case: หากไม่พบข้อความในเอกสาร ให้ถอยกลับไปใช้ Rule-based เพื่อวิเคราะห์จากชื่อไฟล์แทน
    if not sample_text: return classify_document_rule_based(doc)

    file_name = doc.metadata.file_name or ""
    
    # นิยาม Prompt ให้มีความรัดกุม (Strict Instruction) เพื่อลดโอกาสเกิด Hallucination
    prompt = (
        f"Analyze this document content and filename.\n"
        f"Filename: {file_name}\n"
        f"Content Sample (First 4000 chars):\n{sample_text}\n\n"
        f"Classify into exactly one of these types:\n"
        f"{', '.join(CANDIDATE_TYPES)}\n\n"
        f"Guidelines:\n"
        f"- 'contract': Legal agreements, MOUs\n"
        f"- 'manual': User guides, technical specs, handbooks\n"
        f"- 'research_paper': Academic papers, journals\n"
        f"- 'correspondence': Letters, emails, memos\n"
        f"- 'generic': If unsure or general text\n"
        f"\nReply ONLY with the type name (lowercase snake_case)."
    )

    # [แผน A] ประมวลผลผ่าน OpenRouter (Primary Engine)
    client = _get_openai_client()
    if client:
        try:
            res = client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50, temperature=0.0 # ใช้ Temp 0 เพื่อความนิ่งของคำตอบ
            )
            t = res.choices[0].message.content.strip().lower()
            # Clean ผลลัพธ์ให้เหลือเฉพาะตัวอักษรและ Snake_case
            t = re.sub(r"[^a-z_]", "", t)
            if t in CANDIDATE_TYPES: return t
        except Exception as e:
            print(f"[classifier] OpenRouter failed: {e}")

    # [แผน B] ประมวลผลผ่าน Google Gemini (Secondary Fallback)
    google = _get_google_client()
    if google:
        try:
            print("[classifier] 🔄 Using Google Fallback...")
            res = google.generate_content(prompt)
            t = res.text.strip().lower()
            t = re.sub(r"[^a-z_]", "", t)
            if t in CANDIDATE_TYPES: return t
        except Exception as e:
            print(f"[classifier] Google failed: {e}")

    # [แผน C] ระบบคัดกรองด้วยคีย์เวิร์ด (Final Safety Net)
    print("[classifier] AI failed, falling back to rules.")
    return classify_document_rule_based(doc)

# -------------------------------------------------------------------
# PUBLIC INTERFACE (Unified Entrypoint)
# -------------------------------------------------------------------

def classify_document(doc: IngestedDocument, use_llm: bool = True) -> str:
    """
    Interface หลักสำหรับการเรียกใช้งานภายนอก
    - ตัดสินใจเลือกเส้นทางการจำแนกประเภทตามความสมบูรณ์ของข้อมูลและ Configuration
    """
    # กรณีเอกสารไม่มีข้อความ (เช่น Scan ภาพมาแต่ไม่ได้ผ่าน OCR)
    if not doc.texts:
        return classify_document_rule_based(doc)

    # กรณีปิดการใช้งาน LLM เพื่อประหยัด API Cost หรือต้องการความเร็วสูงสุด
    if not use_llm:
        return classify_document_rule_based(doc)

    # โดยปกติจะดำเนินการผ่าน Hybrid LLM Workflow
    return classify_document_with_llm(doc)

# -------------------------------------------------------------------
# CLI TEST (Diagnostic Environment)
# -------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    # จำลอง Path สำหรับการทดสอบ (Mock Environment)
    root = Path("ingested") / "sample"
    meta_path = root / "metadata.json"
    text_path = root / "text.json"

    if not meta_path.exists() or not text_path.exists():
        print("Test files not found. Please run ingestion first.")
    else:
        # โหลดข้อมูลจำลองเพื่อรันกระบวนการ Classification Test
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        texts = json.loads(text_path.read_text(encoding="utf-8"))

        doc = IngestedDocument(
            metadata=DocumentMetadata.from_dict(meta),
            texts=[TextBlock.from_dict(t) for t in texts],
            tables=[],
            images=[],
        )

        print("-" * 50)
        print(f"File: {doc.metadata.filename}")
        print("-" * 50)
        print("Rule-based Result:", classify_document_rule_based(doc))
        print("AI Result (Hybrid):", classify_document(doc, use_llm=True))