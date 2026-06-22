from __future__ import annotations

"""
image_extractor.py (Hybrid Vision Intelligence Edition)

โมดูลสกัดและวิเคราะห์รูปภาพจากเอกสาร (Visual Asset Extraction & Analysis):
- ทำหน้าที่เป็น Orchestrator ในการดึงรูปภาพ (Assets) ออกจากไฟล์ PDF
- [Feature] Hybrid Vision Fallback System: ระบบสร้างคำบรรยายภาพอัตโนมัติ 3 ชั้น
    1) OpenRouter (State-of-the-art VLMs): วิเคราะห์เชิงลึกด้วยโมเดล Vision ระดับสูง
    2) Google Gemini (Low-Latency Backup): ระบบสำรองกรณี API หลักขัดข้องหรือติด Rate Limit
    3) Placeholder/Empty String (Local Fail-safe): ระบบกันตายกรณี Network ขัดข้องทั้งหมด
- แปลงข้อมูลภาพและคำบรรยายให้อยู่ในรูปแบบ ImageBlock Schema เพื่อใช้ในระบบสืบค้น (RAG)
"""

import time
import base64
import os
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv

from .schema import ImageBlock
# [Infrastructure] เรียกใช้ Docling Parser พื้นฐานสำหรับการสกัด Physical Image Files
from .docling_parser import DoclingImageParser

# [Dependencies] ตรวจสอบการติดตั้ง Google Generative AI Library
try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

load_dotenv()

# Global Configuration: กำหนดโมเดล Vision หลักที่ต้องการใช้งาน
VISION_MODEL_NAME = "qwen/qwen2.5-vl-32b-instruct"

# -------------------------------------------------------------------
# Helper: AI Clients & Image Encoding (Data Preparation)
# -------------------------------------------------------------------

def _get_openai_client() -> tuple[Optional[OpenAI], Optional[str]]:
    """
    สร้าง OpenAI Client สำหรับเชื่อมต่อ OpenRouter (Plan A)
    ตรวจสอบความพร้อมของ Environment Variable และ API Key
    """
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")
    
    if not api_key:
        return None, None
        
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, VISION_MODEL_NAME
    except: 
        return None, None

def _get_google_client():
    """
    สร้าง Google Gemini Client สำหรับเป็นระบบสำรอง (Plan B)
    เน้นความเร็วและการประมวลผลข้อมูลในปริมาณมาก
    """
    if not HAS_GOOGLE: return None
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        return None
        
    try:
        genai.configure(api_key=api_key)
        # เลือกใช้ตระกูล Flash เพื่อความคุ้มค่าด้านเวลาและการจัดการ API Quota
        return genai.GenerativeModel('gemini-2.5-flash')
    except: 
        return None

def _encode_image(image_path: Path) -> str:
    """
    แปลงไฟล์รูปภาพ (Binary) ให้อยู่ในรูปแบบ Base64 String
    เพื่อเตรียมส่งไปยังโมเดล Vision ผ่าน REST API
    """
    try:
        # [Security & Integrity] ใช้ Absolute Path เพื่อป้องกันความสับสนของ File Pointer
        abs_path = image_path.resolve()
        with open(abs_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"   ❌ Error encoding image {image_path}: {e}")
        return ""

# -------------------------------------------------------------------
# Logic: Multimodal Captioning (The Hybrid Intelligence)
# -------------------------------------------------------------------

def _generate_caption_hybrid(image_path: Path) -> str:
    """
    หัวใจของระบบวิเคราะห์ภาพ: การจัดการลำดับการเรียกใช้งาน AI (Tiered Processing)
    ลำดับการทำงาน:
    1. Tier 1: OpenRouter (Qwen-VL) -> เน้นคุณภาพสูงสุด (ต้องมี Cooldown 15 วินาทีตามข้อกำหนด API)
    2. Tier 2: Google Gemini -> ทำงานทดแทนกรณี Tier 1 ล้มเหลว (Cooldown สั้น 2 วินาที)
    3. Tier 3: Local Fallback -> คืนค่าว่างเพื่อไม่ให้ระบบหยุดทำงาน (Resilient Design)
    """
    
    # [Pre-flight Check] ตรวจสอบความมีอยู่ของไฟล์ก่อนประมวลผลจริง
    if not image_path.exists():
        print(f"   ❌ Image file missing: {image_path}")
        return ""

    # Structured Prompt สำหรับบังคับรูปแบบการวิเคราะห์ภาพ (Reasoning Guidance)
    prompt = (
        "อธิบายรูปภาพนี้โดยละเอียด: "
        "1. ถ้าเป็นกราฟ/แผนภูมิ ให้บอกชื่อแกน ตัวเลขสำคัญ และแนวโน้ม "
        "2. ถ้าเป็นรูปถ่าย/ไดอะแกรม ให้บอกว่าคืออะไรและมีองค์ประกอบสำคัญอะไรบ้าง "
        "3. ถ้ามีข้อความในภาพ ให้อ่านและสรุปข้อความนั้นมาด้วย "
        "ตอบเป็นภาษาไทย กระชับและได้ใจความ"
    )

    # --- PLAN A: OpenRouter (Primary VLM) ---
    openai_client, model_name = _get_openai_client()
    if openai_client:
        try:
            base64_image = _encode_image(image_path)
            if base64_image:
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                            ],
                        }
                    ],
                    max_tokens=300,
                    timeout=60
                )
                caption = response.choices[0].message.content.strip()
                
                # [Rate Limiting] หน่วงเวลาเพื่อป้องกัน 429 Too Many Requests จากผู้ให้บริการ
                print("   💤 Cooling down Plan A (15s)...")
                time.sleep(15)
                return caption
            
        except Exception as e:
            print(f"   ⚠️ Plan A (OpenRouter) failed: {e}")

    # --- PLAN B: Google Gemini (Secondary Vision) ---
    google_model = _get_google_client()
    if google_model:
        try:
            print(f"   [AI] 🔄 Switching to Plan B: Google Gemini...")
            img = Image.open(image_path)
            response = google_model.generate_content([prompt, img])
            caption = response.text.strip()
            
            # [Performance Optimization] Gemini ตอบสนองเร็ว จึงใช้ Cooldown ขั้นต่ำ
            time.sleep(2)
            return caption
            
        except Exception as e:
            print(f"   ⚠️ Plan B (Google) failed: {e}")

    # --- PLAN C: Local Fallback (The Safety Net) ---
    print("   [AI] ❌ All AI failed. Skipping caption.")
    return ""

# -------------------------------------------------------------------
# Main Extraction & Transformation Pipeline
# -------------------------------------------------------------------

def extract_images(
    file_path: str | Path,
    doc_id: str,
    output_root: str | Path = "ingested",
) -> List[ImageBlock]:
    """
    กระบวนการหลักในการสกัดรูปภาพ (Asset Extraction Pipeline):
    1. แยกรูปภาพออกจาก PDF โดยใช้ Docling Engine
    2. จัดเก็บไฟล์ภาพลงใน Disk ตามโครงสร้างโฟลเดอร์โครงการ
    3. ส่งรูปภาพเข้าสู่ Hybrid AI เพื่อสร้างคำบรรยาย (Captions)
    4. รวบรวมข้อมูลเข้าสู่ ImageBlock Schema เพื่อเป็นส่วนหนึ่งของ IngestedDocument
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # [Directory Management] สร้างโครงสร้างโฟลเดอร์จัดเก็บภาพแยกตาม ID เอกสาร
    output_root = Path(output_root)
    image_dir = output_root / doc_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # ขั้นตอนที่ 1: ดึงรูปภาพทางกายภาพออกจากเอกสาร
    print(f"[image_extractor] Starting Docling extraction for {doc_id}...")
    parser = DoclingImageParser()
    extracted_data = parser.extract_images(str(path), str(image_dir))
    
    image_blocks: List[ImageBlock] = []

    # ขั้นตอนที่ 2: ประมวลผลภาพทีละไฟล์เพื่อทำ Vision Analysis
    for i, item in enumerate(extracted_data):
        # [Integrity] แปลง Path สัมพัทธ์ให้เป็น Absolute Path เพื่อความแม่นยำในการเข้าถึงไฟล์ของ AI
        file_path_on_disk = Path(item["file_path"]).resolve()
        file_name = item["file_name"] 
        
        img_id = f"img_{doc_id}_{item['index']+1:04d}"
        
        print(f"[image_extractor] Generating caption for {file_name}...")
        
        # ขั้นตอนที่ 3: เรียกใช้งานระบบ Hybrid AI เพื่อสร้างความหมายให้รูปภาพ
        caption_text = _generate_caption_hybrid(file_path_on_disk)
        
        # ขั้นตอนที่ 4: การจัดเตรียมข้อมูลในรูปแบบมาตรฐาน (Standardized Schema)
        image_block = ImageBlock(
            id=img_id,
            doc_id=doc_id,
            page=item["page"],
            # ใช้ Relative Path สำหรับการเรียกใช้งานผ่าน UI/Frontend
            file_path=item["file_path"], 
            caption=caption_text, 
            section=None,
            category="figure",
            bbox=item["bbox"],
            extra={
                "source": "docling",
                "original_file_name": file_name,
                "ai_captioned": bool(caption_text),
                "ai_model": "hybrid"
            },
        )
        image_blocks.append(image_block)

    print(f"[image_extractor] Processed {len(image_blocks)} images for {doc_id}.")
    return image_blocks


# -------------------------------------------------------------------
# CLI Entrypoint (Standalone Execution Mode)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Extract images from PDF into ImageBlock list.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--doc-id", help="Document ID (default: stem of file name)")
    parser.add_argument(
        "--output-root",
        default="ingested",
        help="Root folder for saving images (default: 'ingested')",
    )
    args = parser.parse_args()

    pdf_path = args.pdf_path
    doc_id = args.doc_id or Path(pdf_path).stem

    print(f"Extracting images from {pdf_path}...")
    images = extract_images(
        file_path=pdf_path,
        doc_id=doc_id,
        output_root=args.output_root,
    )

    print(f"Extracted {len(images)} images.")