from __future__ import annotations

"""
ocr_extractor.py (Advanced Vision & Local OCR Pipeline)

โมดูลสำหรับการสกัดข้อความจากรูปภาพและไฟล์สแกน (Local OCR Ingestion):
- Digital Text Fallback: ตรวจสอบหา Digital Text Layer ก่อน หากไม่มีจึงเข้าสู่กระบวนการ OCR
- Hybrid Preprocessing: ปรับแต่งภาพด้วย OpenCV เพื่อเพิ่มความแม่นยำ (Grayscale/Contrast Optimization)
- Local Tesseract: สกัดข้อความในเครื่อง 100% ไม่ส่งข้อมูลออก Internet เพื่อ Data Privacy
- AI Refinement: ใช้ Qwen 72B (Local) เกลาคำผิดจากผลลัพธ์ OCR ให้สมบูรณ์แบบ
"""

import io
import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import litellm

from ingestion.config import CUSTOM_API_BASE, CUSTOM_API_KEY, CUSTOM_MODEL_NAME

# 🚨 ชี้พิกัดโปรแกรม Tesseract ในเครื่อง Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# [FIX 1] นิยาม Regex Pattern ที่หายไป สำหรับเช็กว่าเป็นข้อความจริงหรือไม่
_WORD_CHARS_PATTERN = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")

def pdf_page_to_image_bytes(page: fitz.Page, dpi: int = 300) -> bytes:
    """แปลงหน้า PDF เป็น Image Stream เพื่อส่งเข้าสู่กระบวนการ OCR"""
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")

def _clean_text(text: str) -> str:
    """ทำความสะอาดข้อความหลังสกัด (Post-OCR Cleaning)"""
    if not text: return ""
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    # สมานแผลสเปซบาร์ภาษาไทย
    text = re.sub(r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])', '', text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _has_meaningful_text(text: str) -> bool:
    """วิเคราะห์ความหนาแน่นของอักขระเพื่อยืนยันว่าเป็นข้อมูลจริง"""
    if not text: return False
    matches = _WORD_CHARS_PATTERN.findall(text)
    return len(matches) > 5

def _preprocess_image_cv2(image_bytes: bytes) -> bytes:
    """
    [Computer Vision] ปรับแต่งรูปภาพเป็น Grayscale เพื่อให้ Tesseract อ่านภาษาไทยได้แม่นยำขึ้น
    """
    if not image_bytes: return b""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return image_bytes

    # แปลงภาพเป็นโทนสีเทา (Grayscale) ลด Noise 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, encoded_img = cv2.imencode('.png', gray)
    return encoded_img.tobytes()

def refine_text(text: str) -> str:
    """ใช้ Local LLM (Qwen) เกลาคำผิดจาก OCR โดยรักษาความหมายเดิม"""
    if not text or len(text.strip()) < 10: return text
    try:
        response = litellm.completion(
            model=f"openai/{CUSTOM_MODEL_NAME}",
            messages=[
                {"role": "system", "content": "คุณคือผู้เชี่ยวชาญภาษาไทย หน้าที่คือ 'แก้ไขคำสะกดผิด' ที่เกิดจากการสแกน OCR เท่านั้น ห้ามสรุปความ และห้ามแต่งประโยคใหม่เด็ดขาด"},
                {"role": "user", "content": f"ข้อความ OCR ดิบ:\n{text}"}
            ],
            api_base=CUSTOM_API_BASE,
            api_key=CUSTOM_API_KEY,
            temperature=0.0 # ตั้ง 0.0 เพื่อให้ผลลัพธ์คงที่ ไม่มโนเพิ่ม
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️ Refinement Failed: {e}")
        return text

def ocr_page_via_local(image_bytes: bytes) -> str:
    """
    [LOCAL OCR 100%] สแกนข้อความด้วย Tesseract -> เกลาคำผิดด้วย Qwen
    """
    try:
        # [FIX 2] นำภาพไปผ่าน CV2 (ทำขาวดำ) ก่อนส่งให้ Tesseract
        processed_bytes = _preprocess_image_cv2(image_bytes)
        image = Image.open(io.BytesIO(processed_bytes))
        
        # สั่งรัน Tesseract 
        raw_text = pytesseract.image_to_string(image, lang='tha+eng', config='--psm 3')
        
        if raw_text.strip():
            # [FIX 3] เคลียร์อักขระขยะก่อนส่งให้ AI
            cleaned_raw = _clean_text(raw_text)
            return refine_text(cleaned_raw)
        return ""
    except Exception as e:
        print(f"❌ Local OCR Failed: {e}")
        return ""

@dataclass
class OCRDocument:
    texts: List[Dict[str, Any]] = field(default_factory=list)

def ocr_extract_document(pdf_path: str, target_pages: Optional[Set[int]] = None) -> OCRDocument:
    """Workflow Orchestrator สำหรับดึงข้อความทั้งเอกสาร"""
    doc = fitz.open(pdf_path)
    result = OCRDocument()
    
    if target_pages is None:
        print("[OCR] Checking for existing text layer...")
        target_pages = set()
        for idx, page in enumerate(doc):
            raw_text = _clean_text(page.get_text("text") or "")
            if _has_meaningful_text(raw_text):
                print(f"   ✅ Page {idx+1}: Found digital text ({len(raw_text)} chars). Using it.")
                result.texts.append({
                    "page": idx + 1,
                    "content": raw_text,
                    "source": "pdf_text"
                })
            else:
                print(f"   ⚠️ Page {idx+1}: No text found. Marking for Local OCR.")
                target_pages.add(idx + 1)
        
        if not target_pages:
            result.texts.sort(key=lambda x: x["page"])
            doc.close()
            return result

    # ดำเนินการส่งหน้าเป้าหมายเข้าสู่ Local OCR
    if target_pages:
        # [FIX 4] แก้ Log ไม่ให้มีคำว่า API
        print(f"[OCR] Processing {len(target_pages)} image-based pages via Local Tesseract...")
        for idx, page in enumerate(doc):
            page_no = idx + 1
            if page_no in target_pages:
                print(f"   - OCR Scanning Page {page_no}...", end=" ", flush=True)
                image_bytes = pdf_page_to_image_bytes(page)
                ocr_text = ocr_page_via_local(image_bytes)
                
                if ocr_text:
                    print(f"✅ Final Result: {len(ocr_text)} chars.")
                    result.texts.append({
                        "page": page_no,
                        "content": ocr_text,
                        "source": "ocr_local_tesseract" # อัปเดตแหล่งที่มาให้ชัดเจน
                    })
                else:
                    print("❌ Failed or Blank.")

    result.texts.sort(key=lambda x: x["page"])
    doc.close()
    return result