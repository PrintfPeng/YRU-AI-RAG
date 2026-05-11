# backend/dev_test_loader.py

from __future__ import annotations

from pathlib import Path

from backend.services.loader import load_document_bundle
from backend.services.chunking import (
    text_items_to_chunks,
    table_items_to_chunks,
    image_items_to_chunks,
)


def _get_chunk_content(chunk) -> str:
    """
    helper ดึงเนื้อหาหลักของ chunk แบบกันตาย
    - ถ้ามี page_content (เผื่ออนาคตเปลี่ยนเป็น Document) → ใช้อันนี้
    - ถ้ามี content (กรณีเป็น Chunk model ของเรา) → ใช้อันนี้
    """
    if hasattr(chunk, "page_content"):
        return getattr(chunk, "page_content") or ""
    if hasattr(chunk, "content"):
        return getattr(chunk, "content") or ""
    # กันตายสุด ๆ
    return str(chunk)


def _get_chunk_metadata(chunk):
    """
    helper ดึง metadata ของ chunk
    """
    if hasattr(chunk, "metadata"):
        return getattr(chunk, "metadata")
    if hasattr(chunk, "meta"):
        return getattr(chunk, "meta")
    return {}


def main():
    # ปรับ doc_id / path ตรงนี้ถ้าคุณมี doc อื่น
    doc_id = "doc_001"
    base_dir = Path("ingested") / doc_id

    print("=== Dev Test Loader ===")
    print(f"doc_id   : {doc_id}")
    print(f"base_dir : {base_dir}")

    if not base_dir.exists():
        print(f"[ERROR] โฟลเดอร์ {base_dir} ยังไม่มี – ให้ฝั่ง Peng รัน scripts.run_all ก่อน")
        return

    # 1) ลองโหลด DocumentBundle จากโฟลเดอร์ ingested/<doc_id>/
    bundle = load_document_bundle(str(base_dir), doc_id)

    print("\n[METADATA]")
    print(f"  doc_id     : {bundle.metadata.doc_id}")
    print(f"  file_name  : {bundle.metadata.file_name}")
    print(f"  doc_type   : {bundle.metadata.doc_type}")
    print(f"  page_count : {bundle.metadata.page_count}")
    print(f"  ingested_at: {bundle.metadata.ingested_at}")
    print(f"  source     : {bundle.metadata.source}")

    print("\n[COUNTS]")
    print(f"  texts  : {len(bundle.texts)}")
    print(f"  tables : {len(bundle.tables)}")
    print(f"  images : {len(bundle.images)}")

    # 2) แปลงเป็น chunks ด้วยฟังก์ชันเดิมที่ใช้ตอน ingest ลง vector DB
    text_chunks = text_items_to_chunks(bundle)
    table_chunks = table_items_to_chunks(bundle)
    image_chunks = image_items_to_chunks(bundle)

    all_chunks = text_chunks + table_chunks + image_chunks

    print("\n[CHUNKS]")
    print(f"  text chunks : {len(text_chunks)}")
    print(f"  table chunks: {len(table_chunks)}")
    print(f"  image chunks: {len(image_chunks)}")
    print(f"  total chunks: {len(all_chunks)}")

    # 3) แสดงตัวอย่างสัก 1–2 ชิ้นให้เห็นหน้า content + metadata
    if all_chunks:
        sample = all_chunks[0]
        content = _get_chunk_content(sample)
        metadata = _get_chunk_metadata(sample)

        print("\n[SAMPLE CHUNK 1]")
        print("  content :", content[:200].replace("\n", " "))
        print("  metadata:", metadata)

    if len(all_chunks) > 1:
        sample2 = all_chunks[1]
        content2 = _get_chunk_content(sample2)
        metadata2 = _get_chunk_metadata(sample2)

        print("\n[SAMPLE CHUNK 2]")
        print("  content :", content2[:200].replace("\n", " "))
        print("  metadata:", metadata2)

    print("\n=== Dev Test Loader: done ===")


if __name__ == "__main__":
    main()
