from __future__ import annotations

"""
pdf_parser.py (Final Master Orchestrator)

โมดูลหลักสำหรับการประมวลผลไฟล์ PDF (Central Ingestion Engine):
- Orchestration: ควบคุมลำดับการทำงานของ Text, Table และ Image Extractors
- Structural Analysis: วิเคราะห์ Layout (Header/Footer), การเรียงลำดับการอ่าน (Reading Order)
- Semantic Enrichment: ทำการ Tagging ข้อมูลตามเจตนา (Intent) และประเภทเนื้อหา (Warning/Note/Step)
- Vector Integration: จัดเตรียมข้อมูลและนำตารางเข้าสู่ Vector Database (ChromaDB/Pinecone) โดยอัตโนมัติ
- Document Synthesis: รวมรวบผลลัพธ์ทั้งหมดเพื่อสร้างรายงานสรุปในรูปแบบ Smart Markdown (.md)
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Set
import sys
import logging
import re
import statistics

import fitz  # PyMuPDF: High-performance PDF parsing library

from .schema import (
    DocumentMetadata,
    TextBlock,
    IngestedDocument,
    BBox,
)

# ------------------------------------------------------------------
# IMPORTS: External Specialized Extractors & Generators
# ------------------------------------------------------------------
try:
    # นำเข้าโมดูลย่อยที่รับผิดชอบงานเฉพาะทาง (Modular Architecture)
    from .table_extractor import extract_tables, table_to_text
    from .image_extractor import extract_images       # สกัดและทำ Caption รูปภาพด้วย AI
    from .markdown_generator import generate_markdown # แปลง IngestedDocument เป็น Markdown
except ImportError as e:
    # ป้องกันระบบล่ม (Graceful Degradation) หากบางโมดูลย่อยขาดหายไป
    logging.warning(f"[pdf_parser] Dependency import warning: {e}")
    # Define dummy functions เพื่อรักษาความต่อเนื่องของ Pipeline
    if 'extract_tables' not in locals(): extract_tables = lambda *a, **k: []
    if 'table_to_text' not in locals(): table_to_text = lambda x: ""
    if 'extract_images' not in locals(): extract_images = lambda *a, **k: []
    if 'generate_markdown' not in locals(): generate_markdown = lambda *a, **k: None

# ------------------------------------------------------------------
# SYSTEM PATH CONFIGURATION: Vector Store Integration
# ------------------------------------------------------------------
try:
    from backend.services.vector_store import get_vector_store
except ImportError:
    try:
        # แก้ปัญหา Path สำรับการรันจาก CLI หรือสภาพแวดล้อมที่แตกต่าง (Root Path Resolution)
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[1]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        from backend.services.vector_store import get_vector_store
    except ImportError as e:
        raise ImportError(
            f"CRITICAL ERROR: ไม่สามารถเชื่อมต่อกับ 'backend.services.vector_store' ได้\n"
            f"ตรวจสอบโครงสร้างโปรเจกต์และระบบ Import: {e}"
        )

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Config & Text Normalization Helpers
# ==============================================================================
def _generate_doc_id(file_path: Path) -> str:
    """สร้าง Unique Identifier สำหรับเอกสารจากชื่อไฟล์"""
    return file_path.stem.replace(" ", "_").replace("-", "_")

# นิยามอักขระที่ใช้ตรวจสอบความมีอยู่ของเนื้อหา (Alphanumeric + Thai)
_WORD_CHARS_PATTERN = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")

def _clean_text(text: str) -> str:
    """
    [Normalization] ปรับจูนข้อความเบื้องต้น:
    - กำจัด Control Characters และรักษาระเบียบ Newline
    - [Thai Fix] สมานช่องว่างในภาษาไทยเพื่อความถูกต้องในการทำ Semantic Analysis
    """
    if not text: return ""
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    
    # แก้ไขปัญหา Word Segmentation ในภาษาไทยที่มักเกิดจากตัว Parser
    text = re.sub(r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])', '', text)
    
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()

def _is_meaningful_text(text: str) -> bool:
    """ตรวจสอบว่าข้อความมีเนื้อหาเชิงภาษาเพียงพอสำหรับการประมวลผลหรือไม่"""
    if not text: return False
    matches = _WORD_CHARS_PATTERN.findall(text)
    if len(matches) < 2: return False
    return True

def _normalize_section_title(text: str) -> str:
    """ขัดเกลาชื่อหัวข้อ (Section Title) โดยการกำจัดตัวเลขนำหน้าและส่วนเกิน"""
    text = text.strip()
    text = re.sub(r"^(\d+(\.\d+)*|[A-Z])[\.\)]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text[:150]

# ==============================================================================
# 2. Advanced Semantic Analysis (Intent & Entity Recognition)
# ==============================================================================
# คีย์เวิร์ดสำหรับการวิเคราะห์เจตนาของเนื้อหา (Domain-Specific Context)
_INTENT_KEYWORDS = {
    "installation": ["install", "setup", "mounting", "connection", "wiring", "การติดตั้ง", "ต่อสาย"],
    "operation": ["operate", "use", "function", "start", "stop", "การใช้งาน", "วิธีใช้"],
    "troubleshooting": ["error", "fault", "problem", "solution", "fix", "แก้ปัญหา", "อาการเสีย"],
    "maintenance": ["maintain", "clean", "replace", "check", "inspection", "บำรุงรักษา"],
    "safety": ["safety", "warning", "caution", "danger", "hazard", "ความปลอดภัย", "อันตราย"],
    "specification": ["spec", "dimension", "weight", "voltage", "technical data", "ข้อมูลจำเพาะ"]
}

_ENTITY_KEYWORDS = [
    "power button", "led", "lcd", "battery", "fuse", "sensor", "switch", 
    "terminal", "cable", "motor", "pump", "valve", "controller"
]

def _detect_block_type(text: str) -> str:
    """จำแนกประเภทเชิงความหมายของ Text Block (Heading, Warning, Step)"""
    text_upper = text.upper()
    if re.match(r"^(WARNING|CAUTION|DANGER|คำเตือน|ข้อควรระวัง)[:\s]", text_upper):
        return "warning"
    if re.match(r"^(NOTE|NOTICE|IMPORTANT|หมายเหตุ|สำคัญ|ข้อสังเกต)[:\s]", text_upper):
        return "note"
    if re.match(r"^(\d+\.|Step\s+\d+|ขั้นตอนที่\s+\d+|[A-Z]\.)\s", text, re.IGNORECASE):
        return "step"
    return "normal"

def _analyze_intent(text: str, section: str) -> List[str]:
    """วิเคราะห์เจตนา (Intent) ของเนื้อหาเพื่อใช้เป็น Metadata สำหรับการค้นหา (RAG Optimization)"""
    combined = (text + " " + (section or "")).lower()
    intents = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(k in combined for k in keywords):
            intents.append(intent)
    return list(set(intents))

def _extract_entities(text: str) -> List[str]:
    """สกัด Entity สำคัญที่ปรากฏในข้อความ (Keyword Extraction)"""
    text_lower = text.lower()
    found = []
    for entity in _ENTITY_KEYWORDS:
        if entity in text_lower:
            found.append(entity)
    return found

def _determine_answer_scope(block_type: str) -> str:
    """ระบุขอบเขตของเนื้อหา (Answer Scope) เพื่อช่วยในการกรองคำตอบของ AI"""
    if block_type == "step": return "procedure"
    if block_type == "warning": return "warning"
    if block_type == "note": return "note"
    return "general"

# ==============================================================================
# 3. Layout Analysis & Coordinate-Based Sorting
# ==============================================================================
def _detect_header_footer(blocks: List[dict], page_height: float) -> List[dict]:
    """
    [Layout Engine] ระบุตำแหน่ง Header และ Footer โดยใช้เกณฑ์พิกัด (Heuristic Approach):
    - บนสุด 7% ถือเป็น Header Area
    - ล่างสุด 7% ถือเป็น Footer Area
    """
    HEADER_THRESH = page_height * 0.07
    FOOTER_THRESH = page_height * 0.93
    
    for b in blocks:
        x0, y0, x1, y1 = b.get("bbox", (0,0,0,0))
        if "extra" not in b: b["extra"] = {}
        
        is_header = y1 < HEADER_THRESH
        is_footer = y0 > FOOTER_THRESH
        
        b["extra"]["is_header"] = is_header
        b["extra"]["is_footer"] = is_footer
        
        # มาร์คเบื้องต้นว่าเป็น Noise (ข้อความที่ไม่ใช่เนื้อหาหลัก)
        if is_header or is_footer:
            b["extra"]["noise"] = True
            
    return blocks

def _sort_blocks_reading_order(blocks: List[dict]) -> List[dict]:
    """จัดเรียงลำดับการอ่านตามพิกัด Y (บรรทัด) และ X (คอลัมน์) โดยใช้ Grid Alignment"""
    return sorted(blocks, key=lambda b: (int(b["bbox"][1] / 12), b["bbox"][0]))

# ==============================================================================
# 4. Paragraph & Sequence Merge Logic
# ==============================================================================
def _merge_text_blocks(blocks: List[TextBlock]) -> List[TextBlock]:
    """
    [Context Reconstruction] รวม Text Blocks ที่กระจัดกระจายให้เป็นหน่วยข้อมูลที่สมบูรณ์:
    - รวมบรรทัดเข้าเป็น Paragraph ตามขนาดฟอนต์และระยะห่าง
    - รวมขั้นตอนการทำงาน (Step Sequence) เข้าด้วยกันเพื่อรักษาความต่อเนื่องของ Procedure
    """
    if not blocks: return []
    
    merged: List[TextBlock] = []
    current = blocks[0]
    
    for next_block in blocks[1:]:
        curr_type = current.extra.get("block_type", "normal")
        next_type = next_block.extra.get("block_type", "normal")
        
        vertical_dist = next_block.bbox[1] - current.bbox[3]
        
        # เงื่อนไขสำหรับการรวม Procedure (1. -> 2.)
        is_step_sequence = (curr_type in ["step", "step_sequence"] and next_type == "step")
        
        # เงื่อนไขสำหรับการรวมย่อหน้า (font consistency & line spacing)
        curr_font = current.extra.get("font_size", 0)
        next_font = next_block.extra.get("font_size", 0)
        font_diff = abs(curr_font - next_font)
        
        is_paragraph = (
            curr_type == "normal" and next_type == "normal" and
            font_diff < 1.5 and 
            vertical_dist < 15.0 and 
            vertical_dist > -5.0 
        )
        
        same_section = (current.section == next_block.section)

        if same_section and (is_step_sequence or is_paragraph):
            # ดำเนินการ Merge เนื้อหาและ Metadata
            delimiter = "\n" if is_step_sequence else " "
            current.content += delimiter + next_block.content
            
            # ขยาย Bounding Box ให้ครอบคลุมทุกบล็อกที่ถูกรวม
            current.bbox = (
                min(current.bbox[0], next_block.bbox[0]),
                min(current.bbox[1], next_block.bbox[1]),
                max(current.bbox[2], next_block.bbox[2]),
                max(current.bbox[3], next_block.bbox[3]),
            )
            
            if is_step_sequence:
                current.extra["block_type"] = "step_sequence"
                current.extra["answer_scope"] = "procedure"
            
            # รวมรายการ Intent และ Entities จากทุกบล็อก
            curr_intents = set(current.extra.get("intent", []))
            next_intents = set(next_block.extra.get("intent", []))
            current.extra["intent"] = list(curr_intents.union(next_intents))
            
            curr_entities = set(current.extra.get("entities", []))
            next_entities = set(next_block.extra.get("entities", []))
            current.extra["entities"] = list(curr_entities.union(next_entities))
                
        else:
            # เก็บผลลัพธ์และเริ่มต้นบล็อกใหม่
            merged.append(current)
            current = next_block
            
    merged.append(current)
    return merged

# ==============================================================================
# 5. Page-Level Extraction Logic
# ==============================================================================
def _extract_text_blocks_from_page(
    pdf_page: fitz.Page,
    doc_id: str,
    page_number: int,
    start_index: int = 0,
    current_section: Optional[str] = None
) -> Tuple[List[TextBlock], Optional[str]]:
    """กระบวนการสกัดข้อความดิบจากหน้า PDF และการทำ Semantic Tagging เบื้องต้น"""
    
    try:
        # ดึงข้อมูลโครงสร้าง (Dictionary) พร้อม Metadata เชิงลึก
        page_dict = pdf_page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
    except Exception as e:
        logger.warning(f"Page {page_number} dict extraction failed: {e}")
        # กรณีฉุกเฉิน: ใช้ Plain Text Extraction เป็น Fallback
        txt = _clean_text(pdf_page.get_text("text") or "")
        if not _is_meaningful_text(txt):
            return [], current_section
        return [
            TextBlock(
                id=f"txt_{start_index:04d}",
                doc_id=doc_id,
                page=page_number,
                content=txt,
                section=current_section,
                category="fallback",
                bbox=(0.0,0.0,0.0,0.0),
                extra={"noise": False, "block_type": "normal", "intent": [], "entities": []}
            )
        ], current_section

    raw_blocks = page_dict.get("blocks", []) or []
    raw_blocks = _detect_header_footer(raw_blocks, pdf_page.rect.height)
    raw_blocks = _sort_blocks_reading_order(raw_blocks)

    # คำนวณค่ามัธยฐานฟอนต์ของทั้งหน้าเพื่อใช้ในการตรวจหาหัวข้อ (Heading Detection)
    font_sizes = []
    for b in raw_blocks:
        if b.get("type") != 0: continue 
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size"): font_sizes.append(span["size"])
    
    page_median_font = statistics.median(font_sizes) if font_sizes else 10.0

    temp_blocks: List[TextBlock] = []
    current_index = start_index
    active_section = current_section
    
    for block in raw_blocks:
        if block.get("type") != 0: continue # รูปภาพจะถูกจัดการแยกโดย Image Extractor

        # ประกอบคำจาก Spans ในแต่ละ Line ของ Block
        lines = block.get("lines", [])
        spans_text = []
        block_fonts = []
        for line in lines:
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    spans_text.append(t)
                    block_fonts.append(span.get("size", 0))
        
        content = _clean_text(" ".join(spans_text))
        
        # ข้ามข้อมูลขยะ (Noise) หรือข้อมูลที่ไม่มีความหมายเชิงภาษา
        if not _is_meaningful_text(content): continue
        if block.get("extra", {}).get("noise", False): continue

        # [Structural Logic] ตรวจหา Heading จากขนาดฟอนต์และรูปแบบข้อความ
        avg_font = sum(block_fonts)/len(block_fonts) if block_fonts else 0
        is_heading = False
        heading_level = None
        
        if avg_font > page_median_font * 1.2 and len(content) < 200:
            if not re.match(r"^[\d\.\,\s]+$", content): 
                is_heading = True
                heading_level = "H1" if avg_font > page_median_font * 1.5 else "H2"

        block_type = "heading" if is_heading else _detect_block_type(content)
        
        # ปรับปรุงตำแหน่งหัวข้อปัจจุบัน (Section Propagation)
        if is_heading:
            normalized_header = _normalize_section_title(content)
            if normalized_header: active_section = normalized_header

        # เพิ่มเติม Metadata เพื่อใช้ในระบบ RAG และวิเคราะห์คำถาม
        intents = _analyze_intent(content, active_section)
        entities = _extract_entities(content)
        answer_scope = _determine_answer_scope(block_type)

        current_index += 1
        x0, y0, x1, y1 = block.get("bbox", (0,0,0,0))
        
        tb = TextBlock(
            id=f"txt_{current_index:04d}",
            doc_id=doc_id,
            page=page_number,
            content=content,
            section=active_section,
            category=None,
            bbox=(float(x0), float(y0), float(x1), float(y1)),
            extra={
                "font_size": avg_font,
                "is_heading": is_heading,
                "heading_level": heading_level,
                "block_type": block_type,
                "intent": intents,
                "answer_scope": answer_scope,
                "entities": entities
            }
        )
        temp_blocks.append(tb)

    # ผสานบล็อกย่อยเข้าด้วยกันตามตรรกะ Paragraph/Sequence
    merged_blocks = _merge_text_blocks(temp_blocks)
    return merged_blocks, active_section


# ==============================================================================
# 6. Main Parse Function (Integrated End-to-End Pipeline)
# ==============================================================================
def parse_pdf(
    file_path: str | Path,
    doc_type: str = "generic",
    doc_id: Optional[str] = None,
    source: str = "uploaded",
) -> IngestedDocument:
    """
    กระบวนการประมวลผลเอกสารแบบครบวงจร (E2E Document Processing):
    1. สกัด Text พร้อมวิเคราะห์โครงสร้างและ Metadata รายหน้า
    2. สกัดตาราง (Table Extraction) และฝังข้อมูลลง Vector DB ทันทีเพื่อการสืบค้นที่รวดเร็ว
    3. สกัดรูปภาพและสร้างคำบรรยายภาพ (AI Captioning)
    4. ประกอบชิ้นส่วนทั้งหมดเข้าสู่ IngestedDocument Schema
    5. สังเคราะห์รายงานสรุป (Markdown Report) ลงในเครื่อง
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    logger.info(f"[pdf_parser] Processing: {path.name}")
    pdf_doc = fitz.open(path)

    try:
        if doc_id is None:
            doc_id = _generate_doc_id(path)

        metadata = DocumentMetadata(
            doc_id=doc_id,
            file_name=path.name,
            doc_type=doc_type,
            page_count=pdf_doc.page_count,
            ingested_at=datetime.utcnow().isoformat(),
            source=source,
        )

        all_text_blocks: List[TextBlock] = []
        current_index = 0
        current_active_section = None

        # --- ขั้นตอนที่ 1: การสกัดข้อความ (Text Ingestion) ---
        for page_index in range(pdf_doc.page_count):
            page = pdf_doc[page_index]
            
            page_blocks, next_section = _extract_text_blocks_from_page(
                pdf_page=page,
                doc_id=doc_id,
                page_number=page_index + 1,
                start_index=current_index,
                current_section=current_active_section
            )
            
            all_text_blocks.extend(page_blocks)
            current_index += len(page_blocks)
            current_active_section = next_section

        logger.info(f"[pdf_parser] Finished text extraction for {doc_id}: {len(all_text_blocks)} blocks.")

        # --- ขั้นตอนที่ 2: การจัดการข้อมูลตารางและ Vector Storage ---
        logger.info(f"[pdf_parser] Extracting tables for {doc_id}...")
        extracted_tables = extract_tables(
            file_path=path,
            doc_id=doc_id,
            doc_type=doc_type
        )
        
        if extracted_tables:
            logger.info(f"[pdf_parser] Found {len(extracted_tables)} tables. Embedding into Vector Store...")
            try:
                vs = get_vector_store()
                for table in extracted_tables:
                    text = table_to_text(table)
                    
                    # [Normalization] ลบช่องว่างภาษาไทยก่อนบันทึกลงฐานข้อมูล Vector
                    text = re.sub(r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])', '', text)
                    
                    metadata_dict = {
                        "doc_id": table.doc_id,
                        "page": table.page,
                        "source": "table",
                        "category": table.category,
                        "table_id": table.id,
                        "doc_type": doc_type
                    }
                    vs.add_texts(
                        texts=[text],
                        metadatas=[metadata_dict],
                        ids=[f"{table.doc_id}_{table.id}"]
                    )
            except Exception as e:
                logger.error(f"[pdf_parser] Failed to embed tables: {e}")

        # --- ขั้นตอนที่ 3: การสกัดและประมวลผลรูปภาพ (Multimodal) ---
        logger.info(f"[pdf_parser] Extracting images for {doc_id}...")
        ingested_dir = Path("ingested")
        
        try:
            extracted_images = extract_images(
                file_path=path,
                doc_id=doc_id,
                output_root=ingested_dir
            )
        except Exception as e:
            logger.error(f"[pdf_parser] Image extraction failed: {e}")
            extracted_images = []

        # --- ขั้นตอนที่ 4: การประกอบ IngestedDocument (Schema Compilation) ---
        final_doc = IngestedDocument(
            metadata=metadata,
            texts=all_text_blocks,
            tables=extracted_tables,
            images=extracted_images, 
        )

        # --- ขั้นตอนที่ 5: การสร้างรายงานสรุปผล (Report Generation) ---
        doc_output_dir = ingested_dir / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            generate_markdown(final_doc, doc_output_dir)
        except Exception as e:
            logger.error(f"[pdf_parser] Markdown generation failed: {e}")

        return final_doc

    finally:
        pdf_doc.close()

# ==============================================================================
# CLI Testing Interface
# ==============================================================================
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    args = parser.parse_args()

    try:
        doc = parse_pdf(args.pdf_path)
        print(f"✅ Processed {len(doc.texts)} texts, {len(doc.tables)} tables, {len(doc.images)} images.")
    except Exception as e:
        print(f"❌ Error: {e}")