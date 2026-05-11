from __future__ import annotations

"""
cleaner.py

โมดูลหลักสำหรับ Data Sanitization และ Structural Normalization:
- TextBlock: ทำความสะอาด Whitespace, กรอง Noise และเพิ่ม Enrichment Metadata
- TableBlock: ตรวจสอบความสมบูรณ์ของโครงสร้าง (Structural Integrity), Pruning แถว/คอลัมน์ที่ว่างเปล่า

ออกแบบมาให้เป็น Non-destructive process โดยการเก็บค่าตั้งต้นไว้ใน 'extra.cleaning' เพื่อใช้ใน Traceability
"""

from typing import List, Dict, Any
import re

from .schema import TextBlock, TableBlock

# -------------------------------------------------------------------
# Constants & Regex Patterns (สำหรับการประมวลผลข้อความ)
# -------------------------------------------------------------------

# สำหรับลบ Non-printable control characters (ยกเว้น \t, \n, \r)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
# สำหรับจัดการ Invisible artifacts ที่มักติดมาจากกระบวนการ PDF Parsing หรือ OCR
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
NBSP_RE = re.compile(r"\u00A0")

# สำหรับยุบรวม Horizontal whitespace ที่ซ้ำซ้อน
INLINE_WS_RE = re.compile(r"[ \t\r\f\v]+")
# นิยามของ Alphanumeric characters รวมถึงช่วงตัวอักษรภาษาไทย (ก-ฮ, สระ, วรรณยุกต์)
WORD_CHARS_RE = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]")


def _normalize_text(s: str) -> str:
    """
    Standardize ข้อความโดยการกำจัด Artifacts และจัดการ Spacing 
    โดยยังคงรักษาโครงสร้างบรรทัด (Line Breaks) ที่สำคัญไว้
    """
    if not s:
        return ""

    # Strip control characters และ zero-width spaces
    s = CONTROL_CHAR_RE.sub("", s)
    s = ZERO_WIDTH_RE.sub("", s)
    s = NBSP_RE.sub(" ", s)

    # [Thai Language Optimization] ลบช่องว่างส่วนเกินระหว่างตัวอักษรไทย
    # เพื่อแก้ปัญหา Word segmentation ที่ผิดพลาดจากการประมวลผลภาษาไทย
    s = re.sub(r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])', '', s)

    # Collapse ช่องว่างภายในบรรทัดให้เหลือเพียง Single Space
    s = INLINE_WS_RE.sub(" ", s)

    # Trim ช่องว่างที่หัว-ท้ายของแต่ละบรรทัด (Whitespace stripping)
    s = re.sub(r" *\n *", "\n", s)

    # จำกัด Vertical whitespace ไม่ให้เกิน 2 บรรทัดติดต่อกัน เพื่อลดความห่างของย่อหน้า
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def _is_noise_text(s: str) -> bool:
    """
    Heuristic-based filter สำหรับคัดกรองเนื้อหาที่เป็นขยะ (Non-content artifacts)
    คืนค่า True หากข้อความเข้าข่ายเป็น Page number, Separator หรือ Low-information noise
    """
    if not s:
        return True

    # ตรวจสอบความหนาแน่นของตัวอักษรที่มีความหมาย (Semantic Density)
    important = WORD_CHARS_RE.findall(s)
    if len(important) <= 1:
        return True

    # กรองข้อความสั้นที่ไม่มีตัวอักษร (เช่น มีเฉพาะสัญลักษณ์หรือตัวเลขโดดๆ)
    if len(s) <= 3 and not re.search(r"[A-Za-z\u0E00-\u0E7F]", s):
        return True

    # ดักจับรูปแบบ Page numbering (เช่น "- 3 -", "Page 3")
    if re.fullmatch(r"-?\s*\d+\s*-?", s):
        return True

    return False


# -------------------------------------------------------------------
# TextBlock Sanitization
# -------------------------------------------------------------------

def clean_text_blocks(blocks: List[TextBlock]) -> List[TextBlock]:
    """
    ดำเนินการ Batch cleaning สำหรับ TextBlocks:
    - Normalize character encoding และ whitespace
    - Pruning บล็อกที่ระบุว่าเป็น Noise ออกจาก Dataset
    - Update metadata สำหรับการทำ Transformation audit (Cleaning history)
    """
    cleaned: List[TextBlock] = []

    for b in blocks:
        original = b.content or ""
        normalized = _normalize_text(original)

        # Skip บล็อกที่ไม่มีเนื้อหาสำคัญหรือเป็น Noise patterns
        if not normalized or _is_noise_text(normalized):
            continue

        b.content = normalized

        # บันทึกสถานะการทำ Cleaning เพื่อใช้ในกระบวนการ Debug หรือ Audit ภายหลัง
        extra = dict(b.extra or {})
        cleaning_meta: Dict[str, Any] = dict(extra.get("cleaning", {}))
        cleaning_meta.update(
            {
                "original_length": len(original),
                "cleaned_length": len(normalized),
                "removed_chars": max(len(original) - len(normalized), 0),
                "was_noise": False,
            }
        )
        extra["cleaning"] = cleaning_meta
        b.extra = extra

        cleaned.append(b)

    return cleaned


# -------------------------------------------------------------------
# TableBlock Sanitization
# -------------------------------------------------------------------

def _clean_table_cell(cell: Any) -> str:
    """Normalize เนื้อหาภายใน Table cell (Data validation & cleaning)"""
    if cell is None:
        return ""
    return _normalize_text(str(cell))


def clean_table_blocks(tables: List[TableBlock]) -> List[TableBlock]:
    """
    จัดการความถูกต้องของโครงสร้างและคุณภาพข้อมูลใน TableBlocks:
    - Normalize ข้อความในทุก Cell (Header และ Rows)
    - ทำการ Padding แถวเพื่อให้มีจำนวนคอลัมน์เท่ากัน (Structural Consistency)
    - Pruning คอลัมน์และแถวที่ไม่มีข้อมูลสำคัญ (Empty Pruning)
    - บันทึก Metadata การเปลี่ยนแปลงลงใน extra.cleaning
    """
    cleaned_tables: List[TableBlock] = []

    for tb in tables:
        original_header = list(getattr(tb, "header", []) or [])
        original_rows = list(getattr(tb, "rows", []) or [])

        # 1) Sanitize ข้อมูลเชิงอักขระใน Header และ Rows
        header_clean = [_clean_table_cell(h) for h in original_header]
        rows_clean = [[_clean_table_cell(c) for c in (row or [])] for row in original_rows]

        if header_clean or rows_clean:
            # 2) Structural Alignment: คำนวณ Max column width เพื่อทำการ Pad ข้อมูลที่ขาด
            col_count = 0
            if header_clean:
                col_count = max(col_count, len(header_clean))
            for r in rows_clean:
                col_count = max(col_count, len(r))

            header_padded = header_clean + [""] * (col_count - len(header_clean))
            rows_padded = [
                (r + [""] * (col_count - len(r))) for r in rows_clean
            ]

            # 3) Column Pruning: ระบุและกำจัดคอลัมน์ที่ว่างเปล่าทุกแถว (Empty Column Removal)
            keep_col_idx = []
            for idx in range(col_count):
                col_vals = [header_padded[idx]] + [r[idx] for r in rows_padded]
                if any(v.strip() for v in col_vals):
                    keep_col_idx.append(idx)

            header_final = [header_padded[i] for i in keep_col_idx]
            rows_final = [[row[i] for i in keep_col_idx] for row in rows_padded]
        else:
            header_final = header_clean
            rows_final = rows_clean

        # 4) Row Pruning: กำจัดแถวที่ไม่มีข้อมูล (Empty Row Removal)
        rows_final = [r for r in rows_final if any(c.strip() for c in r)]

        tb.header = header_final
        tb.rows = rows_final

        # 5) Metadata Update สำหรับการทำ Tracking และตรวจสอบย้อนกลับ (Traceability)
        extra = dict(tb.extra or {})
        cleaning_meta: Dict[str, Any] = dict(extra.get("cleaning", {}))
        cleaning_meta.update(
            {
                "original_row_count": len(original_rows),
                "cleaned_row_count": len(rows_final),
                "original_header_len": len(original_header),
                "cleaned_header_len": len(header_final),
            }
        )
        extra["cleaning"] = cleaning_meta
        tb.extra = extra

        cleaned_tables.append(tb)

    return cleaned_tables