# backend/services/chunking.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set, Tuple
import re
import hashlib 
from pydantic import BaseModel, Field
from ..models import (
    DocumentBundle,
    TableItem,
    TextBlock,
)

# -------------------------------------------------------------------
# Thai Language Support Initialization
# -------------------------------------------------------------------
try:
    from pythainlp import sent_tokenize
    _HAS_PYTHAINLP = True
except ImportError:
    _HAS_PYTHAINLP = False

# -------------------------------------------------------------------
# Configuration Parameters
# -------------------------------------------------------------------
_TARGET_TOKENS = 300       # อัตราส่วน Token ที่เหมาะสมต่อการแบ่ง Chunk
_MAX_CHUNK_CHARS = 1200    # ขีดจำกัดจำนวนตัวอักษรสูงสุดต่อ 1 Chunk เพื่อป้องกันเกิน Context Window
_CHUNK_OVERLAP = 150       # จำนวนตัวอักษรที่ให้เหลื่อมทับกันระหว่าง Chunk เพื่อรักษาบริบท
_MAX_INTENTS = 5           # จำนวน Intent สูงสุดที่อนุญาตให้มีต่อ 1 Chunk

# -------------------------------------------------------------------
# Intent Priority Configuration
# กำหนดลำดับความสำคัญของเนื้อหา เพื่อให้ Re-ranker จัดอันดับได้แม่นยำขึ้น
# -------------------------------------------------------------------
_INTENT_PRIORITY = {
    "troubleshooting": 5,  # ปัญหา/วิธีแก้ (สำคัญสูงสุด)
    "safety": 5,           # ความปลอดภัย/คำเตือน
    "installation": 4,     # วิธีติดตั้ง/ตั้งค่า
    "financial": 3,        # ข้อมูลการเงิน/ราคา
    "identity": 3,         # ข้อมูลบุคคล/ลายเซ็น
    "reference": 2,        # ข้อมูลอ้างอิง/คำจำกัดความ
    "general": 1,          # เนื้อหาทั่วไป
}

# -------------------------------------------------------------------
# Precompiled Regex Patterns
# ปรับปรุงประสิทธิภาพการทำ Pattern Matching และป้องกัน ReDOS attacks
# -------------------------------------------------------------------

# 1. Intent Keywords (ใช้ระบุจุดประสงค์ของข้อความ)
_KW_INSTALL = r"(?:วิธี|ขั้นตอน|how\s*to|install|setup|การติดตั้ง|วิธีการ)"
_KW_TROUBLESHOOT = r"(?:แก้ปัญหา|error|fail|not\s*working|เสีย|ซ่อม|troubleshoot)"
_KW_SAFETY = r"(?:ความปลอดภัย|warning|danger|ระวัง|ห้าม|อันตราย)"
_KW_REF = r"(?:ความหมาย|คือ|definition|spec|สเปค|คุณลักษณะ)"
_KW_FINANCE = r"(?:ราคา|ค่าใช้จ่าย|เงิน|บาท|cost|price)"
_KW_IDENTITY = r"(?:ผู้|ชื่อ|ลงนาม|อนุมัติ|who|name|signature)"

# 2. Scope Keywords (ใช้ระบุขอบเขต/ลักษณะของเนื้อหา)
_KW_SCOPE_PROC = r"(?:step|ขั้นตอนที่|\d+\.)"
_KW_SCOPE_WARN = r"(?:warning|คำเตือน)"
_KW_SCOPE_TABLE = r"(?:table|ตาราง)"
_KW_SCOPE_EXAMPLE = r"(?:ตัวอย่าง|example|กรณี)"

# 3. Entity Extraction Patterns (ใช้สกัดคำสำคัญเพื่อทำ Indexing)
_RE_MONEY = re.compile(r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:บาท|baht|฿)', re.IGNORECASE)
_RE_YEAR = re.compile(r'(?:ปี\s*)?(\d{4}|พ\.ศ\.\s*\d{4})', re.IGNORECASE)
_RE_THAI_NAME = re.compile(r'(?:นาย|นาง|นางสาว|คุณ|ดร\.|ศ\.|รศ\.|ผศ\.)\s*[\u0E00-\u0E7F]+\s+[\u0E00-\u0E7F]+', re.IGNORECASE)
_RE_HAS_NUM = re.compile(r'\d+')
_RE_QNA = re.compile(r'(?:ถาม|q|question)\s*[:\-]', re.IGNORECASE)

# 4. Content Sanitization Patterns (ใช้ทำความสะอาดและจัดรูปแบบข้อความ)
_RE_SCRIPT = re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL)
_RE_JS_EVENT = re.compile(r" on\w+=", re.IGNORECASE)
_RE_JS_PROTO = re.compile(r"javascript:", re.IGNORECASE)
_RE_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r" {2,}")
_RE_MEANINGFUL = re.compile(r'[\w\u0E00-\u0E7F]{3,}')

# -------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------
class Chunk(BaseModel):
    id: str
    doc_id: str
    doc_type: str
    source: Literal["text", "table", "image"]
    page: Optional[int] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -------------------------------------------------------------------
# Feature: Metadata & Entity Enrichment
# -------------------------------------------------------------------
def _extract_intent_and_entities(text: str, section: str) -> Dict[str, Any]:
    """
    วิเคราะห์ข้อความเพื่อสกัดเจตนา (Intent) และคำสำคัญ (Entities)
    นำไปใช้เป็น Metadata สำหรับการกรอง (Filtering) และเพิ่มน้ำหนัก (Boosting) ใน Vector Database
    """
    text_safe = str(text or "")
    section_safe = str(section or "")
    
    text_lower = text_safe.lower()
    section_lower = section_safe.lower()
    combined = f"{text_lower} {section_lower}"

    # 1. ให้คะแนน Intent โดยอิงจากคีย์เวิร์ดที่พบ
    intent_scores: Dict[str, int] = {}
    
    if re.search(_KW_TROUBLESHOOT, combined):
        intent_scores["troubleshooting"] = 3
    if re.search(_KW_SAFETY, combined):
        intent_scores["safety"] = 3
    if re.search(_KW_INSTALL, combined):
        intent_scores["installation"] = 2
    if re.search(_KW_IDENTITY, combined):
        intent_scores["identity"] = 2
    if re.search(_KW_FINANCE, combined):
        intent_scores["financial"] = 2
    if re.search(_KW_REF, combined):
        intent_scores["reference"] = 1
    
    # จัดเรียง Intent ตามคะแนนและจำกัดจำนวนเพื่อไม่ให้ Metadata มีขนาดใหญ่เกินไป
    intents = []
    if intent_scores:
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        intents = [intent for intent, _ in sorted_intents]
    else:
        intents = ["general"]
    intents = intents[:_MAX_INTENTS]

    # 2. จัดหมวดหมู่ลักษณะการอธิบาย (Scope)
    scope = "general"
    if re.search(_KW_SCOPE_PROC, combined):
        scope = "procedure"
    elif re.search(_KW_SCOPE_WARN, combined):
        scope = "warning"
    elif re.search(_KW_SCOPE_TABLE, combined):
        scope = "tabular"
    elif re.search(_KW_SCOPE_EXAMPLE, combined):
        scope = "example"

    # 3. สกัด Entities เฉพาะเจาะจง (เพื่อความปลอดภัย จำกัดการค้นหาที่ 5000 ตัวอักษรแรก)
    entities = []
    search_text = combined[:5000] 

    try:
        entities.extend([m.group(0) for m in _RE_MONEY.finditer(search_text)])
        entities.extend([m.group(0) for m in _RE_YEAR.finditer(search_text)])
        entities.extend([m.group(0) for m in _RE_THAI_NAME.finditer(search_text)])
    except Exception:
        pass # ป้องกันระบบล่มจาก Regular Expression Error

    unique_entities = sorted(list(set(entities)))[:10]  
    
    # เลือก Intent หลักเพื่อใช้ในการแยกแยะบริบท
    primary_intent = _select_primary_intent(intents)

    return {
        "intent": intents,
        "primary_intent": primary_intent,
        "answer_scope": scope,
        "entities": unique_entities,
        "has_numbers": bool(_RE_HAS_NUM.search(text_safe)),
        "has_names": bool(_RE_THAI_NAME.search(text_safe)),
    }

def _select_primary_intent(intents: List[str]) -> str:
    """
    เลือก Intent หลักอย่างมีระบบ (Deterministic) โดยเรียงจากลำดับความสำคัญ (Priority) 
    และตามด้วยตัวอักษรเพื่อความสม่ำเสมอของข้อมูล
    """
    if not intents:
        return "general"
    return sorted(
        intents,
        key=lambda x: (_INTENT_PRIORITY.get(x, 0), x),
        reverse=True
    )[0]

# -------------------------------------------------------------------
# Feature: Data Sanitization & Normalization
# -------------------------------------------------------------------
def _sanitize_html_content(html_str: str) -> str:
    """ลบสคริปต์และการแทรกแซงที่เป็นอันตราย (XSS/Event Inject) ออกจากโค้ด HTML ตาราง"""
    if not html_str:
        return ""
    try:
        clean = str(html_str)
        clean = _RE_SCRIPT.sub("", clean)
        clean = _RE_JS_EVENT.sub(" data-blocked-event=", clean)
        clean = _RE_JS_PROTO.sub("blocked:", clean)
        return clean.strip()
    except Exception:
        return ""


def _normalize_whitespace(text: str) -> str:
    """
    ล้างช่องว่างส่วนเกินและจัดการอักขระพิเศษ
    รวมถึงแก้ไขปัญหาช่องว่างระหว่างคำภาษาไทย (Thai Word Segmentation Issue) ที่มักเกิดจากระบบ OCR
    """
    if not text:
        return ""
    try:
        s = str(text)
        s = _RE_ZERO_WIDTH.sub("", s)
        s = s.replace("\xa0", " ")
        
        # ค้นหาและลบช่องว่างที่คั่นระหว่างตัวอักษรภาษาไทย เพื่อให้เป็นคำที่สมบูรณ์
        import re
        s = re.sub(r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])', '', s)
        
        s = _RE_MULTI_NEWLINE.sub("\n\n", s)
        s = _RE_MULTI_SPACE.sub(" ", s)
        return s.strip()
    except Exception:
        return ""


def _has_meaningful_text(s: str) -> bool:
    """ตรวจสอบว่าข้อความมีเนื้อหาสาระเพียงพอที่จะจัดเก็บหรือไม่ (ป้องกันการดึงหน้าว่าง/อักขระขยะ)"""
    if not s:
        return False
    s_safe = str(s).strip()
    return bool(_RE_MEANINGFUL.search(s_safe))

def preprocess_thai(text: str) -> str:
    """ทำความสะอาดข้อความและตัดคำด้วย PyThaiNLP"""
    if not text:
        return ""
    clean = _normalize_whitespace(text)
    if _HAS_PYTHAINLP:
        try:
            from pythainlp import word_tokenize
            words = word_tokenize(clean, engine='newmm', keep_whitespace=False)
            return " ".join(words)
        except ImportError:
            return clean
    return clean

# -------------------------------------------------------------------
# Feature: Semantic Text Chunking (การแบ่งก้อนข้อมูลตามความหมาย)
# -------------------------------------------------------------------
def _group_blocks_semantically(blocks: List[TextBlock]) -> List[Dict]:
    """
    จัดกลุ่มข้อความตามบริบท (Intent, Section) และจำกัดขนาดของก้อนข้อมูล (Chunk Size)
    เพื่อรักษาสภาพเนื้อหาไม่ให้ถูกตัดขาดกลางประโยคสำคัญ
    """
    chunks = []
    current_chunk_blocks = []
    current_length = 0
    current_section = None
    current_intent_set: Set[str] = set()
    
    # ระบบ Cache สำหรับเก็บผลลัพธ์ของ Intent ป้องกันการประมวลผลซ้ำซ้อน
    intent_cache: Dict[int, Dict] = {}

    for block in blocks:
        content = _normalize_whitespace(block.content)
        if not content or not _has_meaningful_text(content):
            continue

        block_id = id(block)
        if block_id not in intent_cache:
            intent_cache[block_id] = _extract_intent_and_entities(content, block.section)
        
        block_meta = intent_cache[block_id]
        block_intent_set = set(block_meta["intent"])
        block_len = len(content)

        is_qna = bool(_RE_QNA.search(content))
        
        # ตรวจสอบเงื่อนไขในการแยกเอกสาร (Break Conditions)
        is_new_section = (block.section != current_section) and current_chunk_blocks
        is_major_heading = block.extra.get("heading_level") == "H1"
        is_too_long = (current_length + block_len > _MAX_CHUNK_CHARS)
        
        # ตรวจสอบว่าบริบทเนื้อหามีการเปลี่ยนแปลงอย่างกะทันหันหรือไม่
        intent_changed = False
        if current_chunk_blocks:
            if current_intent_set and block_intent_set:
                if current_intent_set.isdisjoint(block_intent_set):
                    intent_changed = True
            
            last_block = current_chunk_blocks[-1]
            current_primary = intent_cache[id(last_block)]["primary_intent"]
            if current_primary in ["troubleshooting", "safety"]:
                if block_meta["primary_intent"] not in ["troubleshooting", "safety"]:
                    intent_changed = True

        should_break = is_new_section or is_too_long or is_major_heading or intent_changed or is_qna
        
        # ทำการรวมก้อนข้อมูลและเตรียมสร้างก้อนใหม่
        if should_break and current_chunk_blocks:
            chunks.append({
                "blocks": list(current_chunk_blocks),
                "section": current_section,
                "primary_intent": _select_primary_intent(list(current_intent_set))
            })
            current_chunk_blocks = []
            current_length = 0
            current_intent_set = set()

        current_chunk_blocks.append(block)
        current_length += block_len
        current_section = block.section
        current_intent_set.update(block_intent_set)

    # จัดเก็บเนื้อหาส่วนที่เหลือในรอบสุดท้าย
    if current_chunk_blocks:
        chunks.append({
            "blocks": list(current_chunk_blocks),
            "section": current_section,
            "primary_intent": _select_primary_intent(list(current_intent_set))
        })

    return chunks


def _format_chunk_content(group: Dict) -> Tuple[str, Dict]:
    """
    ประกอบข้อความจากกลุ่มที่จัดไว้ และฝัง Metadata ลงในข้อความ
    เพื่อให้ Vector Database นำข้อมูลไปคำนวณ Embedding ได้ครบถ้วน
    """
    blocks: List[TextBlock] = group["blocks"]
    section = group.get("section") or "General"

    if len(section) > 50:
        section = section[:47] + "..."

    raw_text = "\n".join([b.content for b in blocks])
    doc_id = blocks[0].doc_id if blocks else "unknown"
    semantic_meta = _extract_intent_and_entities(raw_text, section)

    content_parts = []
    
    # ฝังคำเตือนพิเศษไว้ในข้อความ เพื่อนำทาง LLM ในกระบวนการ RAG
    if "safety" in semantic_meta["intent"]:
        content_parts.append("⚠️ [ข้อควรระวัง]")
    elif "troubleshooting" in semantic_meta["intent"]:
        content_parts.append("🔧 [การแก้ปัญหา]")
    
    # ฝังคำสำคัญ (Entities) ต่อท้ายเนื้อหา เพื่อเพิ่มความแม่นยำในการค้นหาด้วย Vector DB
    if semantic_meta.get("entities"):
        content_parts.append(f"🔍 [Keywords: {', '.join(semantic_meta['entities'])}]")
    
    page_numbers = set()
    block_types = set()

    for b in blocks:
        prefix = ""
        b_type = str(b.extra.get("block_type", "normal")).lower()
        
        if b_type == "warning":
            prefix = "⚠️ "
        elif b_type == "note":
            prefix = "ℹ️ "

        if prefix:
            content_parts.append(prefix.strip())
        content_parts.append(b.content)

        if b.page:
            page_numbers.add(b.page)
        if b_type:
            block_types.add(b_type)

    full_content = "\n".join(content_parts)
    
    # ป้องกันขนาดข้อความเกินขีดจำกัด
    if len(full_content) > _MAX_CHUNK_CHARS:
        full_content = full_content[:_MAX_CHUNK_CHARS - 50] + "\n...[ตัดทอนเนื้อหา]..."

    representative_page = min(page_numbers) if page_numbers else None
    
    dominant_type = "normal"
    if "warning" in block_types:
        dominant_type = "warning"
    elif "step" in block_types:
        dominant_type = "step"

    metadata = {
        "doc_id": str(doc_id),
        "page": representative_page,
        "pages": sorted(list(page_numbers))[:10],
        "section": section,
        "block_types": sorted(list(block_types))[:5],
        "dominant_block_type": dominant_type,
        "char_count": len(full_content),
        **semantic_meta,
        "source": "text"
    }

    return full_content, metadata


# -------------------------------------------------------------------
# Pipeline 1: Text Ingestion
# -------------------------------------------------------------------
def text_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    chunks: List[Chunk] = []

    valid_blocks = [t for t in bundle.texts if _has_meaningful_text(t.content)]
    if not valid_blocks:
        return chunks

    grouped_chunks = _group_blocks_semantically(valid_blocks)
    seen_hashes = set()

    for group in grouped_chunks:
        content, meta = _format_chunk_content(group)
        if not content.strip():
            continue

        # สร้างลายนิ้วมือทางความหมาย (Semantic Fingerprint) 
        # เพื่อตรวจสอบเนื้อหาซ้ำซ้อน ภายใต้บริบท (Intent + Section) เดียวกัน
        semantic_fingerprint = (
            content 
            + "|" + str(meta.get("primary_intent", "")) 
            + "|" + str(meta.get("section", ""))
        )
        content_hash = hashlib.md5(semantic_fingerprint.encode('utf-8', errors='ignore')).hexdigest()
        
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        chunk_id = f"{meta['doc_id']}::{content_hash[:8]}"
        doc_type = bundle.texts[0].doc_type if bundle.texts and bundle.texts[0].doc_type else "manual"

        chunks.append(
            Chunk(
                id=chunk_id,
                doc_id=str(meta["doc_id"]),
                doc_type=str(doc_type),
                source="text",
                page=meta["page"],
                content=content,
                metadata=meta,
            )
        )

    return chunks


# -------------------------------------------------------------------
# Pipeline 2: Table Chunking (Hybrid Extraction Support)
# -------------------------------------------------------------------
def _normalize_table_extra(item: TableItem) -> Dict[str, Any]:
    """
    จัดรูปแบบโครงสร้าง Metadata ของตารางให้เป็นมาตรฐานกลาง
    ครอบคลุมกรณีการทำ Hybrid Ingestion (จัดเก็บรูปภาพตารางแนบมาด้วย)
    """
    raw_extra = getattr(item, "extra", {}) or {}
    if not isinstance(raw_extra, dict):
        raw_extra = {}

    summary = raw_extra.get("summary")
    if not summary and hasattr(item, "summary"):
         summary = getattr(item, "summary", "")
    
    category = raw_extra.get("category")
    if not category:
        category = getattr(item, "category", None)
    if not category:
        category = "general"
        
    markdown = raw_extra.get("markdown_content") or raw_extra.get("markdown")
    if not markdown and hasattr(item, "markdown"):
        markdown = getattr(item, "markdown", "")
        
    html = raw_extra.get("html_content") or raw_extra.get("html")
    if not html and hasattr(item, "html"):
        html = getattr(item, "html", "")
    
    role = raw_extra.get("role") or getattr(item, "role", None) or ""

    # ดึงข้อมูล Image Path สำหรับแสดงผลในกรณีที่ตารางมีโครงสร้างซับซ้อน (Complex Table)
    image_path = getattr(item, "image_path", None)
    if not image_path:
        image_path = raw_extra.get("image_path")
        
    is_complex = getattr(item, "is_complex", None)
    if is_complex is None:
        is_complex = raw_extra.get("is_complex", False)

    return {
        "summary": str(summary or "").strip(),
        "category": str(category).strip().lower(),
        "markdown_content": str(markdown).strip(), 
        "html_content": str(html).strip(),
        "role": str(role).strip().lower(),
        "image_path": image_path, 
        "is_complex": bool(is_complex) 
    }

def _generate_table_semantic_rows(table: TableItem) -> str:
    """แปลงแถวตารางเป็นข้อความที่มนุษย์ (และ LLM) อ่านเข้าใจง่าย (Semantic Conversion)"""
    if not table.rows or not table.columns:
        return ""
    
    semantic_rows = []
    headers = [str(c) for c in table.columns]
    MAX_ROWS = 15 # จำกัดจำนวนบรรทัดเพื่อป้องกันบริบทล้น
    
    for i, row in enumerate(table.rows[:MAX_ROWS]):
        if not row:
            continue
            
        cells = [_normalize_whitespace(str(c or "")) for c in row]
        if not any(cells):
            continue

        row_parts = []
        for j, cell in enumerate(cells):
            if not cell or len(cell) > 100:
                continue
            col = headers[j] if j < len(headers) else f"Col{j+1}"
            row_parts.append(f"{col}={cell}")
        
        if row_parts:
            semantic_rows.append(" | ".join(row_parts[:5]))

    if len(table.rows) > MAX_ROWS:
        semantic_rows.append(f"... และอีก {len(table.rows) - MAX_ROWS} รายการ")

    return "\n".join(semantic_rows)


def table_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    """
    จัดทำ Chunk ข้อมูลตารางแบบแยกส่วนอิสระ (1 ตาราง = 1 Chunk)
    เพื่อรักษารูปแบบโครงสร้าง (Schema) ให้สอดคล้องกับ Hybrid Ingestion Pipeline
    """
    chunks: List[Chunk] = []
    
    for item in bundle.tables:
        norm_extra = _normalize_table_extra(item)
        
        summary = _normalize_whitespace(norm_extra["summary"])
        category = norm_extra["category"]
        role = norm_extra["role"]
        markdown_raw = _normalize_whitespace(norm_extra["markdown_content"])
        html_raw = norm_extra["html_content"]
        image_path = norm_extra["image_path"]
        is_complex = norm_extra["is_complex"]
        
        item_doc_type = item.doc_type or "manual"
        
        safe_html = _sanitize_html_content(html_raw)
        safe_markdown = markdown_raw[:2000] 
        
        content_parts = []
        
        if item.name:
            clean_name = _normalize_whitespace(item.name)
            content_parts.append(f"📊 {clean_name}")
        
        if category and category != "general":
            content_parts.append(f"ประเภท: {category}")
            
        if summary:
            if len(summary) > 300:
                 summary = summary[:297] + "..."
            content_parts.append(summary)
            
        if item.columns and len(item.columns) <= 10:
             cols = [str(c) for c in item.columns if c]
             if cols:
                 content_parts.append(f"คอลัมน์: {', '.join(cols)}")
        
        # เลือกลำดับความสำคัญของโครงสร้างข้อมูลตาราง
        semantic_rows = _generate_table_semantic_rows(item)
        if markdown_raw:
             display_md = markdown_raw if len(markdown_raw) < 1000 else markdown_raw[:1000] + "..."
             content_parts.append(f"\n[Markdown Data]\n{display_md}")
        elif semantic_rows:
            content_parts.append(f"\n[Row Data]\n{semantic_rows}")
        elif not summary:
            content_parts.append("ตารางข้อมูล (ไม่มีรายละเอียด)")
            
        unified_content = "\n".join(content_parts)
        
        if len(unified_content) > _MAX_CHUNK_CHARS:
             unified_content = unified_content[:_MAX_CHUNK_CHARS - 50] + "\n...[ตัดทอนตาราง]..."

        clean_name_for_intent = _normalize_whitespace(item.name) if item.name else ''
        combined_for_intent = f"{clean_name_for_intent}\n{summary}\n{unified_content}"
        semantic_meta = _extract_intent_and_entities(combined_for_intent, category)
        
        # ผูกคำสำคัญแนบท้ายเนื้อหาตาราง
        if semantic_meta.get("entities"):
            unified_content += f"\n🔍 [Keywords: {', '.join(semantic_meta['entities'])}]"

        metadata = {
            "table_id": item.id,
            "doc_id": str(item.doc_id),
            "page": item.page,
            "columns": list(item.columns) if item.columns else [],
            "has_summary": bool(summary),
            "html_content": safe_html,       # เก็บโครงสร้าง HTML สำหรับการแสดงผลหลังบ้านเท่านั้น
            "markdown_content": safe_markdown, 
            "category": category,
            "role": role,
            "html_trusted": False,
            "source": "table",
            "image_path": image_path,     # Path รูปภาพที่จะนำไปแสดงในช่องแชท (Hybrid Layout)
            "is_complex": is_complex,     
            **semantic_meta,
        }

        chunks.append(
            Chunk(
                id=f"{item.doc_id}::table::{item.id}",
                doc_id=str(item.doc_id),
                doc_type=str(item_doc_type),
                source="table",
                page=item.page,
                content=unified_content,
                metadata=metadata,
            )
        )

    return chunks


# -------------------------------------------------------------------
# Pipeline 3: Image Chunking (Image-to-Text Enrichment)
# -------------------------------------------------------------------
def image_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    chunks: List[Chunk] = []
    
    for item in bundle.images:
        content = _normalize_whitespace(item.caption or "")
        if not content or not _has_meaningful_text(content):
            continue

        item_doc_type = item.doc_type or "manual"
        semantic_meta = _extract_intent_and_entities(content, "Image")

        clean_path = str(item.file_path or "").replace("\\", "/")
        
        # จัดรูปแบบคำอธิบายภาพ เพื่อให้ Vector DB สกัดความหมายได้แม่นยำ
        formatted_content = (
            f"🖼️ [Image Info]\n"
            f"Path: {clean_path}\n"
            f"Page: {item.page or '?'}\n"
            f"Description: {content}"
        )
        
        if semantic_meta.get("entities"):
            formatted_content += f"\n🔍 [Keywords: {', '.join(semantic_meta['entities'])}]"

        chunks.append(
            Chunk(
                id=f"{item.doc_id}::image::{item.id}",
                doc_id=str(item.doc_id),
                doc_type=str(item_doc_type),
                source="image",
                page=item.page,
                content=formatted_content,
                metadata={
                    "image_id": item.id,
                    "file_path": str(item.file_path or ""),
                    "doc_id": str(item.doc_id),
                    "page": item.page,
                    "source": "image",
                    **semantic_meta,
                },
            )
        )
    
    return chunks