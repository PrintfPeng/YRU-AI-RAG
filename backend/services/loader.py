# backend/services/loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..models import (
    DocumentBundle,
    ImageItem,
    Metadata,
    TableItem,
    TextItem,
)


def _load_json(path: Path) -> dict | list:
    """
    ฟังก์ชันช่วยเหลือ (Helper) สำหรับโหลดข้อมูลจากไฟล์ JSON
    
    Args:
        path (Path): เส้นทางไปยังไฟล์ JSON
        
    Raises:
        FileNotFoundError: หากไม่พบไฟล์ตามเส้นทางที่ระบุ
        
    Returns:
        dict | list: ข้อมูลที่ถูกแปลงกลับจากโครงสร้าง JSON
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _load_json_if_exists(path: Path) -> dict | list | None:
    """
    ฟังก์ชันช่วยเหลือ (Helper) สำหรับโหลดไฟล์ JSON แบบมีเงื่อนไข (Optional)
    
    Args:
        path (Path): เส้นทางไปยังไฟล์ JSON
        
    Returns:
        dict | list | None: คืนค่าข้อมูล JSON หากพบไฟล์ หากไม่พบจะคืนค่า None เพื่อป้องกันระบบล่ม
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def load_document_bundle(base_dir: str, doc_id: str) -> DocumentBundle:
    """
    รวบรวมและโหลดข้อมูลทั้งหมดของเอกสาร 1 ชุด (รหัส doc_id) จากโฟลเดอร์ผลลัพธ์ (Ingested Directory)
    เพื่อจัดกลุ่มเป็น DocumentBundle สำหรับส่งต่อเข้าสู่กระบวนการ Chunking และ Vectorization

    โครงสร้างไดเรกทอรีที่คาดหวัง (อ้างอิงจาก Pipeline ขาเข้า):
        ingested/<doc_id>/
          ├── metadata.json
          ├── text.json
          ├── table.json
          ├── image.json
          ├── text_clean.json (optional)
          ├── table_clean.json (optional)
          ├── text_enriched.json (optional)
          ├── table_normalized.json (optional)
          └── mapping.json (optional)

    กลยุทธ์การเลือกไฟล์ (Fallback Priority Logic):
    ระบบจะเลือกดึงข้อมูลจากไฟล์ที่มีความสมบูรณ์สูงสุด (Enriched/Normalized) เป็นลำดับแรก
    - Text Priority:  text_enriched.json > text_clean.json > text.json
    - Table Priority: table_normalized.json > table_clean.json > table.json
    
    Args:
        base_dir (str): พาธหลักของโฟลเดอร์เอกสาร (เช่น "ingested/doc_xxxx")
        doc_id (str): รหัสเอกสารที่ต้องการโหลด
        
    Returns:
        DocumentBundle: วัตถุที่รวม Metadata, Texts, Tables และ Images ทั้งหมดเข้าด้วยกัน
    """

    base_path = Path(base_dir)

    # ----------------------------------------------------
    # 1) โหลดและตรวจสอบ Metadata 
    # ----------------------------------------------------
    metadata_raw = _load_json(base_path / "metadata.json")
    metadata = Metadata(**metadata_raw)

    # ตรวจสอบความสอดคล้องของ ID ระหว่างพารามิเตอร์ที่รับมากับข้อมูลในไฟล์
    # หากไม่ตรงกัน ให้ยึดถือ ID จากไฟล์ Metadata เป็นหลัก (Source of Truth)
    if metadata.doc_id != doc_id:
        print(
            f"[WARN] metadata.doc_id ({metadata.doc_id}) mismatch requested doc_id ({doc_id}) "
            f"-> Proceeding with metadata.doc_id as primary identifier."
        )
        doc_id = metadata.doc_id

    # ----------------------------------------------------
    # 2) โหลดข้อมูลข้อความ (Text Extraction) พร้อมใช้ลำดับความสำคัญ (Fallback Priority)
    # ----------------------------------------------------
    text_enriched_path = base_path / "text_enriched.json"
    text_clean_path = base_path / "text_clean.json"
    text_raw_path = base_path / "text.json"

    text_list_raw = None
    text_source_name = None

    if text_enriched_path.exists():
        text_list_raw = _load_json(text_enriched_path)
        text_source_name = "text_enriched.json"
    elif text_clean_path.exists():
        text_list_raw = _load_json(text_clean_path)
        text_source_name = "text_clean.json"
    else:
        # กรณีไม่มีการทำ Cleaning หรือ Enrichment จะใช้ข้อมูลดิบจาก text.json (Required File)
        text_list_raw = _load_json(text_raw_path)
        text_source_name = "text.json"

    print(f"[loader] Text Extraction: Using '{text_source_name}' for doc_id={doc_id}")

    # แนบข้อมูล doc_id และ doc_type กลับเข้าไปในทุกๆ ก้อนข้อความ (Item)
    # เพื่อป้องกันข้อมูลสูญหายกรณี Pipeline ต้นทางไม่ได้แนบมาให้
    for item in text_list_raw:
        item.setdefault("doc_id", metadata.doc_id)
        item.setdefault("doc_type", metadata.doc_type)

    texts: List[TextItem] = [TextItem(**item) for item in text_list_raw]

    # ----------------------------------------------------
    # 3) โหลดข้อมูลตาราง (Table Extraction) พร้อมใช้ลำดับความสำคัญ (Fallback Priority)
    # ----------------------------------------------------
    table_norm_path = base_path / "table_normalized.json"
    table_clean_path = base_path / "table_clean.json"
    table_raw_path = base_path / "table.json"

    table_list_raw = None
    table_source_name = None

    if table_norm_path.exists():
        table_list_raw = _load_json(table_norm_path)
        table_source_name = "table_normalized.json"
    elif table_clean_path.exists():
        table_list_raw = _load_json(table_clean_path)
        table_source_name = "table_clean.json"
    else:
        table_list_raw = _load_json(table_raw_path)
        table_source_name = "table.json"

    print(f"[loader] Table Extraction: Using '{table_source_name}' for doc_id={doc_id}")

    for item in table_list_raw:
        item.setdefault("doc_id", metadata.doc_id)
        item.setdefault("doc_type", metadata.doc_type)

    tables: List[TableItem] = [TableItem(**item) for item in table_list_raw]

    # ----------------------------------------------------
    # 4) โหลดข้อมูลรูปภาพ (Image Extraction)
    # ----------------------------------------------------
    image_raw_path = base_path / "image.json"
    
    # รูปภาพยังไม่มีกระบวนการ Normalized จึงโหลดจากไฟล์ดิบโดยตรง
    image_list_raw = _load_json(image_raw_path)

    for item in image_list_raw:
        item.setdefault("doc_id", metadata.doc_id)
        item.setdefault("doc_type", metadata.doc_type)

    images: List[ImageItem] = [ImageItem(**item) for item in image_list_raw]

    # ----------------------------------------------------
    # 5) ประกอบร่างเป็น DocumentBundle
    # ----------------------------------------------------
    bundle = DocumentBundle(
        metadata=metadata,
        texts=texts,
        tables=tables,
        images=images,
    )
    
    return bundle