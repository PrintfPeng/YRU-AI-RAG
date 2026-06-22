# backend/models.py
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Tuple, Dict, Union

from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------------
# Core Type Definitions
# --------------------------------------------------------

# กำหนดชนิดข้อมูล BBox (Bounding Box) สำหรับอ้างอิงพิกัดบนหน้าเอกสาร (x0, y0, x1, y1)
BBox = Tuple[float, float, float, float]


# --------------------------------------------------------
# Extracted Data Models (ตัวแทนข้อมูลที่สกัดได้จากเอกสาร)
# --------------------------------------------------------

class TextItem(BaseModel):
    """
    โมเดลตัวแทนข้อมูลข้อความ (Text Block) 1 ส่วนที่สกัดได้จากเอกสาร
    รองรับการจัดเก็บข้อความทั่วไปและข้อมูลอภิพันธุ์ (Rich Metadata) จากระบบ Parser
    """
    # อนุญาตให้มีฟิลด์นอกเหนือจากที่กำหนดได้ เพื่อความยืดหยุ่นในการรับข้อมูล
    model_config = ConfigDict(extra="allow") 

    id: str
    doc_id: str
    page: int

    section: Optional[str] = None
    content: str

    # พิกัด Bounding Box รองรับทั้งรูปแบบ Dictionary และ List เพื่อให้เข้ากันได้กับ Parser หลายประเภท
    bbox: Any | None = None

    category: Optional[str] = None

    # กำหนดให้ doc_type เป็น Optional เนื่องจากในขั้นแรก Parser อาจยังไม่ได้ระบุ 
    # โดยระบบ Loader จะเป็นผู้ทำหน้าที่เติมข้อมูลส่วนนี้ในภายหลัง
    doc_type: Optional[str] = None
    
    # ฟิลด์รองรับข้อมูล Metadata เพิ่มเติมแบบยืดหยุ่น (Dynamic Metadata)
    extra: Dict[str, Any] = Field(default_factory=dict)


# สร้าง Alias ให้ TextBlock ชี้ไปที่ TextItem เพื่อความเข้ากันได้ของระบบ (Backward Compatibility)
# และป้องกัน Type Error ระหว่างกระบวนการโหลดข้อมูลจาก JSON มายังระบบ Chunking
TextBlock = TextItem


class TableItem(BaseModel):
    """
    โมเดลตัวแทนข้อมูลตาราง (Table Data) 1 ชุดที่สกัดได้จากเอกสาร
    """
    model_config = ConfigDict(extra="allow")

    id: str
    doc_id: str
    page: int

    name: Optional[str] = None
    section: Optional[str] = None
    category: Optional[str] = None

    # ใช้ default_factory เพื่อป้องกันข้อผิดพลาดกรณีไม่มีข้อมูลคอลัมน์
    columns: List[str] = Field(default_factory=list) 
    
    # รองรับข้อมูลในเซลล์ทุกประเภท (Any) พร้อมกำหนดค่าเริ่มต้นเป็นลิสต์ 2 มิติ
    rows: List[List[Any]] = Field(default_factory=list) 

    bbox: Any | None = None
    doc_type: Optional[str] = None
    
    # ฟิลด์รองรับข้อมูล Metadata เพิ่มเติม (เช่น โครงสร้าง HTML, Markdown หรือ Image Path)
    # ตามที่กระบวนการ Hybrid Extraction ในโมดูล Chunking ต้องการ
    extra: Dict[str, Any] = Field(default_factory=dict)


class ImageItem(BaseModel):
    """
    โมเดลตัวแทนข้อมูลรูปภาพ (Image Data) 1 ภาพที่สกัดได้จากเอกสาร
    """
    model_config = ConfigDict(extra="allow")

    id: str
    doc_id: str
    page: int

    file_path: str
    caption: Optional[str] = None

    section: Optional[str] = None
    category: Optional[str] = None

    bbox: Any | None = None
    doc_type: Optional[str] = None
    
    # ฟิลด์รองรับข้อมูล Metadata เพิ่มเติมที่เกี่ยวข้องกับรูปภาพ (เช่น ผลลัพธ์จากการทำ OCR)
    extra: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------
# Document-Level Models (ข้อมูลระดับภาพรวมเอกสาร)
# --------------------------------------------------------

class Metadata(BaseModel):
    """
    โมเดลเก็บข้อมูลอภิพันธุ์ (Metadata) ภาพรวมของเอกสารทั้งฉบับ
    """
    model_config = ConfigDict(extra="allow")

    doc_id: str
    file_name: str

    # กำหนดให้ doc_type เป็น Optional เพื่อรองรับกรณีที่เอกสารยังไม่ถูกจัดประเภทในขั้นตอนแรกเริ่ม
    doc_type: Optional[str] = None

    page_count: int

    # รองรับเวลาทั้งรูปแบบ datetime object และ ISO8601 string 
    # เพื่อความยืดหยุ่นในการรับข้อมูลจากหลายแหล่ง (Pydantic จะทำการ Parse ให้อัตโนมัติ)
    ingested_at: Union[datetime, str]
    source: str
    
    # ฟิลด์รองรับข้อมูล Metadata ระดับเอกสารเพิ่มเติมแบบไดนามิก
    extra: Dict[str, Any] = Field(default_factory=dict)


# สร้าง Alias เพื่อความเข้ากันได้กับโครงสร้างของระบบ PDF Parser รุ่นก่อนหน้า
DocumentMetadata = Metadata


# --------------------------------------------------------
# Core DTO: DocumentBundle
# --------------------------------------------------------

class DocumentBundle(BaseModel):
    """
    โมเดลรวมข้อมูลทั้งหมดของเอกสาร 1 ฉบับ (ผูกด้วย doc_id เดียวกัน)
    ประกอบด้วย:
    - metadata (ข้อมูลภาพรวมเอกสาร)
    - texts (รายการข้อความทั้งหมด)
    - tables (รายการตารางทั้งหมด)
    - images (รายการรูปภาพทั้งหมด)

    ทำหน้าที่เป็นโครงสร้างข้อมูลกลาง (Data Transfer Object - DTO) 
    สำหรับส่งต่อให้กระบวนการ Chunking, Embeddings และ RAG
    """

    metadata: Metadata
    
    # ใช้ TextItem เป็นชนิดข้อมูลหลักเพื่อให้ตรงกับรูปแบบที่ถูกสกัดมาจากระบบ Parser โดยตรง
    texts: List[TextItem] = Field(default_factory=list) 
    tables: List[TableItem] = Field(default_factory=list)
    images: List[ImageItem] = Field(default_factory=list)

# สร้าง Alias เพื่อความเข้ากันได้ของชื่อคลาสกับส่วนอื่นๆ ในระบบ Pipeline
IngestedDocument = DocumentBundle