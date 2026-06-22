from __future__ import annotations

"""
schema.py (Document Data Model & Schema Definition)

โมดูลกำหนดโครงสร้างข้อมูลมาตรฐาน (Canonical Data Model) ของระบบ:
- นิยาม Dataclasses สำหรับจัดเก็บผลลัพธ์จากการสกัดเอกสาร (Texts, Tables, Images)
- ระบบ Robust Serialization: รองรับการแปลงข้อมูลไป-กลับระหว่าง Object และ Dictionary (JSON-ready)
- Schema Tolerance: ออกแบบให้รองรับข้อมูล Legacy และ Unknown keys เพื่อป้องกันระบบพังเมื่อมีการขยายฟิลด์ในอนาคต
- Type Safety: ใช้ Typing และ Validation Helpers เพื่อควบคุมคุณภาพของข้อมูล (Data Integrity)
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, Type, TypeVar, Union

# =============================================================================
# GLOBAL TYPES & VALIDATION HELPERS
# =============================================================================

# BBox Type: พิกัดล้อมรอบวัตถุในรูปแบบ (x1, y1, x2, y2) อ้างอิงตาม PDF Page Coordinates
BBox = Tuple[float, float, float, float]

def _safe_bbox(bbox_raw: Any) -> Optional[BBox]:
    """
    ตรวจสอบและสกัดพิกัด Bounding Box จากข้อมูลดิบ (Input Validation)
    คืนค่าเป็น Tuple ของ float 4 ตำแหน่ง หรือ None หากข้อมูลไม่ถูกต้องตามมาตรฐาน
    """
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        try:
            return (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
        except (ValueError, TypeError):
            return None
    return None

def _safe_list(value: Any) -> List[Any]:
    """Helper สำหรับรับประกันว่าข้อมูลที่ส่งมาจะเป็น List (Fail-soft to empty list)"""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []

def _safe_dict(value: Any) -> Dict[str, Any]:
    """Helper สำหรับรับประกันว่าข้อมูลที่ส่งมาจะเป็น Dictionary"""
    if isinstance(value, dict):
        return value
    return {}

def _normalize_str(value: Any) -> Optional[str]:
    """
    ทำ String Normalization: ตัดช่องว่างส่วนเกินและเปลี่ยนเป็นตัวพิมพ์เล็ก
    คืนค่า None หากเป็นค่าว่างหรือข้อมูลไม่ใช่ String (Semantic Consistency)
    """
    if isinstance(value, str):
        s = value.strip().lower()
        if s:
            return s
    return None

def _normalize_enum(value: Any, valid_set: set[str], default: str) -> str:
    """ตรวจสอบความถูกต้องของข้อมูลเทียบกับเซตของค่าที่อนุญาต (Enum-like Validation)"""
    norm = _normalize_str(value)
    if norm in valid_set:
        return norm
    return default

# =============================================================================
# DocumentMetadata (Document-Level Identity)
# =============================================================================

@dataclass
class DocumentMetadata:
    """
    Metadata หลักของเอกสารต้นฉบับ:
    - ใช้สำหรับระบุตัวตน (Identification) และใช้ในกระบวนการ Indexing/Filtering
    - เก็บข้อมูลแหล่งที่มา (Source) และเวลาที่ประมวลผล (Ingestion Timestamp)
    """
    doc_id: str
    file_name: str
    doc_type: str                  # ประเภทเอกสาร (เช่น "invoice", "bank_statement")
    page_count: int
    ingested_at: str               # เวลาประมวลผลในรูปแบบ ISO 8601
    source: str = "uploaded"       # ช่องทางการนำเข้าข้อมูล

    def to_dict(self) -> Dict[str, Any]:
        """แปลง Object เป็น Dictionary สำหรับจัดเก็บลงฐานข้อมูล"""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["DocumentMetadata"], data: Dict[str, Any]) -> "DocumentMetadata":
        """สร้าง Metadata Object จากข้อมูล Dictionary (Deserialization)"""
        d = _safe_dict(data)
        return cls(
            doc_id=str(d.get("doc_id", "")),
            file_name=str(d.get("file_name", "")),
            doc_type=_normalize_str(d.get("doc_type")) or "generic",
            page_count=int(d.get("page_count", 0) or 0),
            ingested_at=str(d.get("ingested_at", "")),
            source=_normalize_str(d.get("source")) or "uploaded",
        )


# =============================================================================
# TextBlock (Granular Content Container)
# =============================================================================

@dataclass
class TextBlock:
    """
    หน่วยจัดเก็บข้อความระดับบล็อก (Text Segment):
    - เก็บข้อมูลพิกัด (BBox) และตำแหน่งหน้า เพื่อใช้ในการสร้าง Reading Order
    - รองรับการติด Semantic Tags (Section, Category, Role) สำหรับงาน RAG
    """
    id: str
    doc_id: str
    page: int
    content: str
    section: Optional[str] = None      # เช่น "header", "footer"
    category: Optional[str] = None     # ป้ายกำกับสำหรับ RAG เช่น "legal_clause"
    role: Optional[str] = None         # บทบาทของข้อความ เช่น "title", "paragraph"
    bbox: Optional[BBox] = None
    extra: Dict[str, Any] = field(default_factory=dict) # Metadata เพิ่มเติมจากการประมวลผล

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["TextBlock"], data: Dict[str, Any]) -> "TextBlock":
        d = _safe_dict(data)
        return cls(
            id=str(d.get("id", "")),
            doc_id=str(d.get("doc_id", "")),
            page=int(d.get("page", 1) or 1),
            content=str(d.get("content", "")),
            section=_normalize_str(d.get("section")),
            category=_normalize_str(d.get("category")),
            role=_normalize_str(d.get("role")),
            bbox=_safe_bbox(d.get("bbox")),
            extra=_safe_dict(d.get("extra")),
        )

# =============================================================================
# TableBlock (Structured & Hybrid Representation)
# =============================================================================

@dataclass
class TableBlock:
    """
    โมเดลจัดเก็บตารางที่มีความยืดหยุ่นสูง (Hybrid Table Model):
    - รองรับทั้งข้อมูลโครงสร้าง (Columns/Rows) และรูปภาพ (Image Path) สำหรับตารางที่ซับซ้อน
    - เก็บข้อมูลความเชื่อมั่น (Trust Level) และระบุว่าโครงสร้างเป็นแบบ Lossy หรือไม่
    - ออกแบบมาให้รองรับการสกัดจากทั้ง Programmatic Engines และ Vision-based AI
    """

    # --- Identity & Physical Location ---
    id: str
    doc_id: str
    page: int
    
    # --- Semantic & Classification Metadata ---
    name: Optional[str] = None        # ชื่อตาราง (ถ้าพบในเอกสาร)
    section: Optional[str] = None 
    category: Optional[str] = None
    role: Optional[str] = None

    # --- Primary Data Structures ---
    columns: List[str] = field(default_factory=list) # ส่วนหัวตาราง (Header)
    rows: List[List[Any]] = field(default_factory=list) # ข้อมูลในแต่ละแถว

    # --- Output Representations (LLM & UI Ready) ---
    markdown: Optional[str] = None     # สำหรับส่งให้ LLM วิเคราะห์
    html_content: Optional[str] = None # สำหรับเรนเดอร์บน Web UI

    # --- Hybrid Handling (Image Support) ---
    image_path: Optional[str] = None   # เส้นทางไฟล์รูปภาพตารางที่ถูก Crop ออกมา
    is_complex: bool = False           # แฟล็กระบุว่าเป็นฟอร์มหรือตารางที่มีโครงสร้างซับซ้อน

    # --- Extraction Audit Trail (การตรวจสอบย้อนกลับ) ---
    source: str = "unknown"            # Engine ที่ใช้สกัด (เช่น Camelot, Docling)
    method: Optional[str] = None       # อัลกอริทึมที่ใช้ (เช่น Lattice, Stream)
    numeric_trust: str = "unknown"     # ระดับความเชื่อมั่นของข้อมูลตัวเลข (High, Medium, Low)
    
    # --- Structural Integrity Flags ---
    structured_available: bool = False # ระบุว่าข้อมูลแบบ Columns/Rows พร้อมใช้งานหรือไม่
    raw_available: bool = False        # ระบุว่ามีข้อมูลแบบ Markdown/HTML หรือไม่
    structure_lossy: bool = False      # ระบุว่าโครงสร้างตารางอาจมีการผิดเพี้ยน (กรณีใช้ Vision)

    # --- Spatial & Metadata Extensions ---
    bbox: Optional[BBox] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # --- Property Aliases (Backward Compatibility) ---
    @property
    def header(self) -> List[str]:
        """Alias สำหรับเข้าถึง Columns ผ่านคีย์ 'header' เพื่อความสะดวกใน Legacy Code"""
        return self.columns

    @header.setter
    def header(self, value: List[str]) -> None:
        self.columns = list(value or [])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ข้อมูลทั้งหมดเป็น Dictionary (ไม่รวม Computed Properties)"""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["TableBlock"], data: Dict[str, Any]) -> "TableBlock":
        """
        Robust Deserializer: โหลดข้อมูลตารางอย่างปลอดภัย
        - รองรับ Alias 'header' -> 'columns'
        - Harvest Unknown Keys: รวบรวมฟิลด์ที่ไม่รู้จักเข้าสู่ 'extra' เพื่อรักษาความสมบูรณ์ของข้อมูล
        - Inference Logic: ทำการคาดเดาสถานะของ Flags หากไม่มีการระบุมาโดยตรง
        """
        d = _safe_dict(data)

        # 1. จัดการข้อมูลส่วนเกิน (Schema Forward Compatibility)
        extra_data = _safe_dict(d.get("extra")).copy()
        known_fields = {
            "id", "doc_id", "page", "name", "section", "category", "role",
            "columns", "rows", "header", "markdown", "html_content", 
            "image_path", "is_complex", "source", "method", "numeric_trust",
            "structured_available", "raw_available", "structure_lossy",
            "bbox", "extra"
        }
        for k, v in d.items():
            if k not in known_fields:
                extra_data[k] = v

        # 2. จัดการ Mapping ของ Columns/Header
        raw_columns = d.get("columns") or d.get("header")
        columns: List[str] = [str(c) for c in _safe_list(raw_columns)]

        # 3. จัดการโครงสร้าง Rows (Soft-fail on invalid format)
        rows_raw = _safe_list(d.get("rows"))
        rows: List[List[Any]] = [list(r) for r in rows_raw if isinstance(r, (list, tuple))]

        # 4. Resolve Content Aliases (Markdown/HTML)
        markdown_val = d.get("markdown") or d.get("markdown_content") or extra_data.get("markdown")
        html_val = d.get("html_content") or d.get("html") or extra_data.get("html_content")
            
        source_val = _normalize_str(d.get("source")) or "unknown"
        numeric_trust_val = _normalize_enum(
            d.get("numeric_trust"), {"high", "medium", "low", "unknown"}, "unknown"
        )

        # 5. Logical Inference สำหรับ Flags
        structured_avail = d.get("structured_available")
        if structured_avail is None:
            structured_avail = bool(columns and rows)

        raw_avail = d.get("raw_available")
        if raw_avail is None:
            raw_avail = bool(markdown_val or html_val)

        lossy = d.get("structure_lossy")
        if lossy is None:
            lossy = (source_val == "vision" or numeric_trust_val == "low")

        return cls(
            id=str(d.get("id", "")),
            doc_id=str(d.get("doc_id", "")),
            page=int(d.get("page", 1) or 1),
            name=d.get("name"),
            section=_normalize_str(d.get("section")),
            category=_normalize_str(d.get("category")),
            role=_normalize_str(d.get("role")),
            columns=columns,
            rows=rows,
            markdown=markdown_val,
            html_content=html_val,
            image_path=d.get("image_path"),
            is_complex=bool(d.get("is_complex", False)),
            source=source_val,
            method=_normalize_str(d.get("method")),
            numeric_trust=numeric_trust_val,
            structured_available=bool(structured_avail),
            raw_available=bool(raw_avail),
            structure_lossy=bool(lossy),
            bbox=_safe_bbox(d.get("bbox")),
            extra=extra_data,
        )

# =============================================================================
# ImageBlock (Visual Asset Metadata)
# =============================================================================

@dataclass
class ImageBlock:
    """
    หน่วยจัดเก็บข้อมูลรูปภาพที่สกัดจากเอกสาร:
    - เก็บเส้นทางไฟล์ (File Path) และคำบรรยายภาพ (AI-generated Caption)
    - รองรับ Alias 'image_path' เพื่อความยืดหยุ่นในการเขียนโค้ด
    """
    id: str
    doc_id: str
    page: int
    file_path: str                 # เส้นทางจัดเก็บไฟล์รูปภาพ (Local/Storage)
    caption: Optional[str] = None  # คำอธิบายภาพ (สกัดจาก AI หรือ Metadata)
    section: Optional[str] = None
    category: Optional[str] = None # ประเภทรูปภาพ (เช่น "chart", "logo", "figure")
    role: Optional[str] = None     # บทบาทเชิงความหมาย เช่น "visual_data"
    bbox: Optional[BBox] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def image_path(self) -> str:
        return self.file_path

    @image_path.setter
    def image_path(self, value: str) -> None:
        self.file_path = value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type["ImageBlock"], data: Dict[str, Any]) -> "ImageBlock":
        d = _safe_dict(data)
        file_path = d.get("file_path") or d.get("image_path") or ""

        return cls(
            id=str(d.get("id", "")),
            doc_id=str(d.get("doc_id", "")),
            page=int(d.get("page", 1) or 1),
            file_path=str(file_path),
            caption=d.get("caption"),
            section=_normalize_str(d.get("section")),
            category=_normalize_str(d.get("category")),
            role=_normalize_str(d.get("role")),
            bbox=_safe_bbox(d.get("bbox")),
            extra=_safe_dict(d.get("extra")),
        )


# =============================================================================
# IngestedDocument (Root Aggregate Container)
# =============================================================================

TIngested = TypeVar("TIngested", bound="IngestedDocument")

@dataclass
class IngestedDocument:
    """
    คอนเทนเนอร์หลัก (Root Aggregate) ของเอกสารที่ผ่านการประมวลผลสมบูรณ์แล้ว:
    - รวบรวม Metadata และส่วนประกอบย่อยทั้งหมด (Text, Table, Image) เข้าด้วยกัน
    - ทำหน้าที่เป็น Schema กลางสำหรับการส่งต่อข้อมูลระหว่างโมดูล (Data Interchange Format)
    - มีระบบ Versioning เพื่อรองรับการเปลี่ยนแปลงโครงสร้างข้อมูลในอนาคต
    """
    metadata: DocumentMetadata
    texts: List[TextBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    images: List[ImageBlock] = field(default_factory=list)
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes ทั้งเอกสารให้อยู่ในรูปแบบ Deep Dictionary"""
        return {
            "metadata": self.metadata.to_dict(),
            "texts": [t.to_dict() for t in self.texts],
            "tables": [tb.to_dict() for tb in self.tables],
            "images": [im.to_dict() for im in self.images],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls: Type[TIngested], data: Dict[str, Any]) -> TIngested:
        """ประกอบ Object เอกสารคืนจาก Dictionary (Full Reconstitution)"""
        d = _safe_dict(data)
        return cls(
            metadata=DocumentMetadata.from_dict(d.get("metadata", {})),
            texts=[TextBlock.from_dict(t) for t in _safe_list(d.get("texts"))],
            tables=[TableBlock.from_dict(tb) for tb in _safe_list(d.get("tables"))],
            images=[ImageBlock.from_dict(im) for im in _safe_list(d.get("images"))],
            schema_version=str(d.get("schema_version", "1.0")),
        )