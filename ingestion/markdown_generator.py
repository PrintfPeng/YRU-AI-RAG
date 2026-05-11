from __future__ import annotations

"""
markdown_generator.py (Final UI/UX Rendering Engine)

โมดูลสำหรับการจัดทำรายงานในรูปแบบ Markdown (Document Synthesis):
- Reading Order Restoration: เรียงลำดับเนื้อหาตามพิกัดจริง (Page, Y, X) เพื่อรักษาลำดับการอ่าน
- Thai Language Sanitization: จัดการช่องว่างและอักขระพิเศษในภาษาไทยให้ถูกต้องตามหลักไวยากรณ์
- Semantic Formatting: แปลง AI Role (เช่น Warning, Legal, Q&A) ให้เป็น Markdown Elements ที่สื่อความหมาย
- Rich Media Integration: รองรับการแสดงผลตารางพร้อมคำสรุป และรูปภาพแบบ Clickable Relative Links
- Metadata Enrichment: แสดงข้อมูลภาพรวมของเอกสารที่หัวกระดาษ (Header)
"""

import re
from pathlib import Path
from typing import List, Any
from dataclasses import dataclass
from .schema import IngestedDocument, TextBlock, TableBlock, ImageBlock

@dataclass
class RenderItem:
    """
    Data Structure สำหรับใช้ในกระบวนการจัดลำดับเนื้อหา (Sorting Buffer)
    เก็บตำแหน่งพิกัดและประเภทข้อมูลเพื่อใช้ในการทำ Reading Order Restoration
    """
    page: int
    y0: float
    x0: float
    type: str  # 'text', 'table', 'image'
    content: Any

def _clean_text_for_markdown(text: str) -> str:
    """
    [Data Cleaning] ปรับปรุงคุณภาพข้อความก่อนเข้าสู่กระบวนการ Rendering:
    - กำจัด Non-printable characters และ Null bytes
    - แก้ไขปัญหาช่องว่างเกินความจำเป็นระหว่างตัวอักษรภาษาไทย
    """
    if not text: return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    # [Thai Optimization] ลบ Space ที่คั่นกลางระหว่างอักขระไทย (มักพบในงาน OCR)
    text = re.sub(r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])', '', text)
    return text.strip()

def _format_text_block(block: TextBlock) -> str:
    """
    [Semantic Rendering] แปลง TextBlock เป็น Markdown ตามบทบาท (Role) ที่ AI วิเคราะห์ได้:
    - รองรับโครงสร้างเอกสารระดับ Professional (Header, Legal Clauses, Warning Notes)
    - รองรับรูปแบบการตอบโต้ (Q&A Format)
    """
    text = _clean_text_for_markdown(block.content)
    if not text: return ""

    role = block.extra.get("role", "normal_text")
    
    if role == "title":
        return f"\n# {text}\n"
    elif role == "section_header":
        return f"\n## {text}\n"
    elif role == "legal_clause":
        return f"\n> ⚖️ **{text}**\n"
    elif role == "warning_note":
        return f"\n> ⚠️ **คำเตือน:** {text}\n"
    elif role == "instruction_step":
        return f"1. {text}"
    elif role == "list_item":
        return f"- {text}"
    elif role == "page_meta":
        return f"\n*(Page {text})*\n"
    elif role == "qna_question":
        return f"\n**Q: {text}**"
    elif role == "qna_answer":
        return f"\n**A: {text}**\n"
    
    return text

def _format_table_block(block: TableBlock) -> str:
    """
    [Table Synthesis] จัดรูปแบบตารางพร้อมข้อมูลประกอบ:
    - เพิ่มคำบรรยายสรุปตารางจาก AI (Summary) เพื่อความรวดเร็วในการทำความเข้าใจ
    - Render ข้อมูลในรูปแบบ Markdown Table Standard
    """
    md_lines = []
    
    if block.name:
        md_lines.append(f"\n### 📊 ตาราง: {block.name}")
    
    summary = block.extra.get("summary")
    if summary:
        md_lines.append(f"> *AI Summary: {summary}*")
    
    if block.markdown:
        md_lines.append(block.markdown)
    else:
        md_lines.append("*(ตารางไม่มีข้อมูล)*")
    
    md_lines.append("\n")
    return "\n".join(md_lines)

def _format_image_block(block: ImageBlock) -> str:
    """
    [Media Rendering] จัดรูปแบบรูปภาพให้มีปฏิสัมพันธ์ (Interactive):
    - สร้าง Link รูปภาพแบบ Clickable ที่อ้างอิงแบบ Relative Path
    - แสดงคำบรรยายภาพ (AI Caption) และลิงก์ไปยังไฟล์ต้นฉบับ
    """
    md_lines = []
    
    # ดึงชื่อไฟล์จาก Path เพื่อสร้าง Link สัมพัทธ์สำหรับใช้งานบนเว็บหรือ Markdown Viewer
    image_name = Path(block.file_path).name
    caption = block.caption or "Image"
    
    # กำหนด Path สำหรับ Folder รูปภาพที่สกัดไว้
    rel_path = f"images/{image_name}"
    
    # Render Clickable Image: คลิกที่รูปเพื่อเปิดไฟล์ภาพขนาดเต็ม
    md_lines.append(f"\n[![{caption}]({rel_path})]({rel_path})")
    
    # แสดงคำบรรยายใต้ภาพ (Captioning)
    if block.caption:
        md_lines.append(f"\n*รูปที่: {block.caption}*")
    
    md_lines.append(f"> 📂 File: [`{image_name}`]({rel_path})\n")
    
    return "\n".join(md_lines)

def generate_markdown(doc: IngestedDocument) -> str:
    """
    [Main Pipeline] กระบวนการสร้างไฟล์ Markdown ทั้งระบบ:
    1. รวบรวม Blocks ทุกประเภท (Text, Table, Image) เข้าสู่ Sorting Buffer
    2. ทำการ Sort ข้อมูลตาม Page -> Y -> X เพื่อกู้คืนลำดับการอ่าน (Reading Order Restoration)
    3. สร้างส่วนหัวของเอกสาร (Document Header) จาก Metadata
    4. วนลูปสร้างเนื้อหาแยกตามหน้าและประเภทข้อมูล
    """
    items: List[RenderItem] = []
    
    # ขั้นตอนที่ 1: รวบรวมข้อมูลลงใน Buffer พร้อมพิกัด Bounding Box
    for t in doc.texts:
        y0 = t.bbox[1] if t.bbox else 0.0
        x0 = t.bbox[0] if t.bbox else 0.0
        items.append(RenderItem(page=t.page, y0=y0, x0=x0, type='text', content=t))
        
    for tb in doc.tables:
        y0 = tb.bbox[1] if tb.bbox else 0.0
        x0 = tb.bbox[0] if tb.bbox else 0.0
        items.append(RenderItem(page=tb.page, y0=y0, x0=x0, type='table', content=tb))
        
    for im in doc.images:
        y0 = im.bbox[1] if im.bbox else 0.0
        x0 = im.bbox[0] if im.bbox else 0.0
        items.append(RenderItem(page=im.page, y0=y0, x0=x0, type='image', content=im))
        
    # ขั้นตอนที่ 2: จัดเรียงลำดับเนื้อหาให้ถูกต้องตามการอ่านจริง
    items.sort(key=lambda x: (x.page, x.y0, x.x0))
    
    # ขั้นตอนที่ 3: สร้าง Document Header และ Metadata Overview
    md_content = []
    
    # [Integrity Check] ตรวจสอบความถูกต้องของ Metadata Attributes (file_name, ingested_at)
    file_name = getattr(doc.metadata, "file_name", "Unknown")
    ingested_at = getattr(doc.metadata, "ingested_at", "Unknown")

    md_content.append(f"# 📄 เอกสาร: {file_name}") 
    md_content.append(f"- **ประเภท:** {doc.metadata.doc_type}")
    md_content.append(f"- **วันที่ประมวลผล:** {ingested_at}")
    md_content.append("---\n")
    
    # ขั้นตอนที่ 4: วนลูป Render เนื้อหาลงใน Markdown Content
    current_page = 0
    
    for item in items:
        # ระบบจัดการ Page Break: แสดงหน้าใหม่เมื่อมีการเปลี่ยนหมายเลขหน้า
        if item.page > current_page:
            current_page = item.page
            md_content.append(f"\n\n--- \n## 📑 Page {current_page}\n\n")
            
        # เลือกใช้ Formatter ตามประเภทของข้อมูล
        if item.type == 'text':
            md_content.append(_format_text_block(item.content))
        elif item.type == 'table':
            md_content.append(_format_table_block(item.content))
        elif item.type == 'image':
            md_content.append(_format_image_block(item.content))
            
    # คืนค่าผลลัพธ์ในรูปแบบ String เพื่อจัดเก็บเป็นไฟล์ .md ต่อไป
    return "\n".join(md_content)