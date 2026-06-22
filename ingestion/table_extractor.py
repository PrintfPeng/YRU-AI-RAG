from __future__ import annotations

"""
table_extractor.py (Advanced Hybrid Table Extraction Engine)

โมดูลสกัดข้อมูลตารางประสิทธิภาพสูง (Multi-strategy Table Parser):
- Strategy A: Camelot (Programmatic Extraction) - ให้ความแม่นยำของข้อมูลตัวเลข 100% (High Trust)
- Strategy B: Vision AI (Qwen-VL/Gemini) - สำหรับตารางที่ไม่มีเส้นขอบ หรือ Layout ซับซ้อนที่ Parser ปกติอ่านไม่ได้
- Orchestration Logic: ใช้ระบบ Prioritization โดยให้ความสำคัญกับ Camelot ก่อน หากไม่พบจึงใช้ Vision AI 
- Intelligence Layer: ระบบ AI 3 ชั้นสำหรับทำ Summarization, Classification และ Data Validation
- Data Connectivity: รองรับการแปลงตารางเป็นข้อความ (Textualization) เพื่อใช้ในกระบวนการ Vector Embedding
"""

import time
import io
import re
import hashlib 
import base64
import os
from pathlib import Path
from typing import List, Optional, Any, Tuple
from html.parser import HTMLParser

# External Libraries for Document Processing
import camelot
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv

from .schema import TableBlock 

# [AI Infrastructure] OpenAI/OpenRouter Integration
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# [AI Infrastructure] Google Generative AI Fallback Integration
try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

load_dotenv()

# -------------------------------------------------------------------
# Configuration & Domain Constraints
# -------------------------------------------------------------------

MIN_ROWS = 2 # จำนวนแถวขั้นต่ำที่ถือว่าเป็นตารางที่มีนัยสำคัญ
MIN_COLS = 2 # จำนวนคอลัมน์ขั้นต่ำ
MAX_HEADER_SCAN_ROWS = 3

SCHEMA_VERSION = "tableblock_v1"
PARSER_VERSION = "2026-01-Hybrid"

DEFAULT_TIMEOUT = 120.0

# Keywords สำหรับการแยกตารางย่อย (Sub-table splitting)
HEADER_PATTERNS = [
    r"ประวัติการศึกษา",
    r"ประวัติการอบรม",
    r"ความสามารถทางภาษา",
    r"รายการ",
    r"ลำดับที่",
]

# AI Model Selection
VISION_MODEL = "qwen/qwen2.5-vl-32b-instruct"
TEXT_MODEL = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")


# -------------------------------------------------------------------
# Data Sanitization & Encoding Helpers
# -------------------------------------------------------------------

def _clean_thai_text(text: Any) -> str:
    """
    [Thai Language Optimization] จัดการปัญหาการขึ้นบรรทัดใหม่กลางคำ (Mid-word newline)
    ซึ่งมักพบในตาราง PDF เพื่อให้ข้อความมีความต่อเนื่องเชิงความหมาย
    """
    if text is None:
        return ""
    text = str(text).strip()
    # ลบ Newline ที่อยู่ระหว่างอักษรไทยก-ฮ (รวมสระและวรรณยุกต์)
    text = re.sub(r'(?<=[\u0E00-\u0E7F])\s*[\r\n]+\s*(?=[\u0E00-\u0E7F])', '', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _has_meaningful_text(s: str) -> bool:
    """ตรวจสอบว่าเซลล์มีข้อมูลที่เป็นอักขระ (Alphanumeric) หรือไม่"""
    if s is None: return False
    s = str(s).strip()
    return bool(s) and any(ch.isalnum() for ch in s)

def _compute_row_content_hash(rows: list[list[str]]) -> str:
    """สร้าง Digital Fingerprint ของเนื้อหาตารางเพื่อใช้ในกระบวนการ Deduplication (ป้องกันตารางซ้ำ)"""
    row_content = "".join(["".join(map(str, r)) for r in rows])
    row_content = re.sub(r"[\s\u200b]+", "", row_content).lower()
    return hashlib.md5(row_content.encode('utf-8')).hexdigest()

def _pil_image_to_base64(image: Image.Image) -> str:
    """แปลง PIL Image เป็น Base64 สำหรับส่งไปยัง Vision API"""
    b = io.BytesIO()
    image.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode("utf-8")


# -------------------------------------------------------------------
# Visual Intelligence: HTML Table Parser
# -------------------------------------------------------------------

class SimpleTableParser(HTMLParser):
    """
    Custom HTML Parser สำหรับประมวลผลผลลัพธ์จาก Vision AI:
    - สกัด Header และ Body จากโครงสร้าง <table>
    - ตรวจสอบความซับซ้อนของตาราง (เช่น Colspan/Rowspan)
    - Normalize จำนวนคอลัมน์ในทุกแถวให้เท่ากันเพื่อความสมบูรณ์ของข้อมูล
    """
    def __init__(self):
        super().__init__()
        self.in_table = False; self.in_thead = False; self.in_tbody = False
        self.in_tr = False; self.in_th = False; self.in_td = False
        self.current_text = []; self.headers = []; self.current_row = []
        self.rows = []
        self.has_complex_body = False
        self.has_complex_header = False

    def handle_starttag(self, tag, attrs):
        if tag == 'table': self.in_table = True
        elif tag == 'thead': self.in_thead = True
        elif tag == 'tbody': self.in_tbody = True
        elif tag == 'tr': self.in_tr = True; self.current_row = []
        elif tag in ('th', 'td'):
            if tag == 'th': self.in_th = True
            else: self.in_td = True
            self.current_text = []
            
            # ตรวจสอบการผสานเซลล์ (Merge Cells) ซึ่งมีผลต่อความแม่นยำของโครงสร้าง
            attr_dict = {k.lower(): v for k, v in attrs}
            try:
                r, c = int(attr_dict.get('rowspan', '1')), int(attr_dict.get('colspan', '1'))
                if r > 1 or c > 1:
                    if self.rows: self.has_complex_body = True
                    else: 
                        if r > 1: self.has_complex_header = True
            except: pass

    def handle_endtag(self, tag):
        if tag == 'table': self.in_table = False
        elif tag == 'thead': self.in_thead = False
        elif tag == 'tbody': self.in_tbody = False
        elif tag == 'tr':
            self.in_tr = False
            if not self.headers: self.headers = self.current_row
            else:
                if self.current_row: self.rows.append(self.current_row)
            self.current_row = []
        elif tag in ('th', 'td'):
            self.in_th = False; self.in_td = False
            self.current_text.append(''.join(self.current_text).strip())
            self.current_text = []

    def handle_data(self, data):
        if self.in_th or self.in_td: self.current_text.append(data)

    def get_table_data(self) -> Tuple[list[str], list[list[str]]]:
        """คืนค่าข้อมูลตารางที่ผ่านการจัดลำดับและ Normalize จำนวนคอลัมน์แล้ว"""
        cols = self.headers
        data = self.rows
        if not cols and data: cols = data[0]; data = data[1:]
        if not cols: return [], []
        
        norm_rows = []
        expected = len(cols)
        for r in data:
            if len(r) > expected: norm_rows.append(r[:expected])
            elif len(r) < expected: norm_rows.append(r + [""]*(expected-len(r)))
            else: norm_rows.append(r)
        return cols, norm_rows

def parse_html_table(html: str) -> Tuple[list[str], list[list[str]], bool, bool]:
    """Helper สำหรับการแปลง HTML Table String เป็น Python Data Structures"""
    parser = SimpleTableParser()
    try:
        parser.feed(html)
        cols, rows = parser.get_table_data()
        
        # กรองตารางที่เป็นขยะหรือซับซ้อนเกินกว่าจะแสดงผลแบบ Structured Data ได้
        if cols and not rows: return [], [], True, parser.has_complex_header
        if parser.has_complex_body: return [], [], True, parser.has_complex_header

        cols = [_clean_thai_text(c) for c in cols]
        rows = [[_clean_thai_text(cell) for cell in row] for row in rows]
        return cols, rows, parser.has_complex_body, parser.has_complex_header
    except: return [], [], True, False


# -------------------------------------------------------------------
# Representation Formatters (MD/HTML)
# -------------------------------------------------------------------

def table_to_markdown(columns: list[str], rows: list[list[Any]]) -> str:
    """แปลงตารางเป็นรูปแบบ Markdown สำหรับการใช้งานใน LLM Prompt หรือ Reports"""
    if not columns: return ""
    lines = ["| " + " | ".join(map(str, columns)) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    for r in rows:
        padded = list(r) + [""]*(len(columns)-len(r))
        lines.append("| " + " | ".join(map(str, padded[:len(columns)])) + " |")
    return "\n".join(lines)

def table_to_html(columns: list[str], rows: list[list[Any]]) -> str:
    """แปลงตารางเป็นรูปแบบ HTML พร้อมคลาสสำหรับ UI Rendering (Tailwind-compatible)"""
    if not columns: return ""
    h = ['<table class="min-w-full text-sm text-left border-collapse border">']
    h.append('<thead class="bg-gray-100"><tr>' + ''.join(f'<th class="px-4 py-2 border">{c}</th>' for c in columns) + '</tr></thead><tbody>')
    for r in rows:
        padded = list(r) + [""]*(len(columns)-len(r))
        h.append('<tr>' + ''.join(f'<td class="px-4 py-2 border">{c}</td>' for c in padded[:len(columns)]) + '</tr>')
    h.append('</tbody></table>')
    return ''.join(h)


# -------------------------------------------------------------------
# AI Infrastructure & Intelligence Logic
# -------------------------------------------------------------------

def _get_llm_client() -> Optional[OpenAI]:
    """สร้าง Client สำหรับ Tier 1 AI (OpenRouter)"""
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")
    if not api_key: return None
    if OpenAI is None: return None
    try:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=DEFAULT_TIMEOUT)
    except: return None

def _get_google_client():
    """สร้าง Client สำหรับ Tier 2 AI (Google Gemini)"""
    if not HAS_GOOGLE: return None
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')
    except: return None


def _summarize_table(client: OpenAI, markdown_table: str, is_html: bool = False) -> str:
    """
    [Semantic Summary] ใช้ AI ในการสรุปสาระสำคัญของตาราง (Executive Summary):
    - ช่วยให้ระบบค้นหา (RAG) เข้าใจบริบทของตารางได้เร็วขึ้น
    - ทำงานแบบ Hybrid Fallback (OpenRouter -> Google)
    """
    truncated = _truncate_html_safely(markdown_table) if is_html else markdown_table[:4000]
    prompt = f"สรุปใจความสำคัญของตารางนี้สั้นๆ (ไม่เกิน 3 บรรทัด):\nข้อมูล:\n{truncated}"

    # PLAN A: OpenRouter (Primary Model)
    if client:
        try:
            res = client.chat.completions.create(
                model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=150, temperature=0.1, timeout=60.0
            )
            time.sleep(15) # Cooldown เพื่อป้องกัน Rate limit สำหรับระบบสเกลใหญ่
            return res.choices[0].message.content.strip()
        except Exception: pass
    
    # PLAN B: Google Gemini (Reliable Fallback)
    google = _get_google_client()
    if google:
        try:
            res = google.generate_content(prompt)
            time.sleep(2)
            return res.text.strip()
        except Exception: pass
    
    return ""

def _classify_category_with_llm(client: OpenAI, text_sample: str) -> str:
    """จำแนกหมวดหมู่ของตาราง (Classification) เพื่อใช้ในการทำ Metadata Tagging"""
    input_text = f"table_context: {text_sample}".strip()
    if len(input_text) < 50: input_text += " generic_table_hint"
    
    prompt = (
        f"Classify table category from: slogan_holder, parade, fancy, student_council, equipment, budget, schedule, staff, generic_table\n"
        f"Data: '{input_text[:1000]}'\nReply only category name (snake_case)."
    )
    
    if client:
        try:
            res = client.chat.completions.create(
                model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=50, temperature=0.0, timeout=30.0
            )
            cat = re.sub(r"[^a-z_]", "", res.choices[0].message.content.strip().lower())
            return cat if cat else "generic_table"
        except Exception: pass

    google = _get_google_client()
    if google:
        try:
            res = google.generate_content(prompt)
            cat = re.sub(r"[^a-z_]", "", res.text.strip().lower())
            return cat if cat else "generic_table"
        except Exception: pass
    
    return "generic_table"

def _extract_table_with_vision(client: OpenAI, image: Image.Image) -> str:
    """
    [Vision Extraction Layer] สกัดตารางจากภาพ (Screenshot):
    - แปลงรูปภาพเป็น HTML Table เพื่อรักษาความสัมพันธ์เชิงพื้นที่ (Spatial Relationships)
    - ใช้สำหรับตารางที่วิธี Programmatic (Camelot) เข้าถึงไม่ได้
    """
    if not client: return ""
    prompt = "Extract table to HTML. Use only <table>, <thead>, <tbody>, <tr>, <th>, <td> tags. No markdown."
    try:
        b64 = _pil_image_to_base64(image)
        res = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
            max_tokens=2000, timeout=DEFAULT_TIMEOUT
        )
        content = getattr(res.choices[0].message, "content", "") or ""
        html = content.replace("```html", "").replace("```", "").strip()
        
        if "<table" in html:
            time.sleep(15)
            return html
        return ""
    except Exception as e:
        print(f"[table_extractor] Vision extraction failed: {e}")
        return ""


# -------------------------------------------------------------------
# Logic: Data Processing & Structure Transformation
# -------------------------------------------------------------------

def _dataframe_to_columns_rows(df) -> Tuple[list[str], list[list[Any]]]:
    """แปลง Pandas DataFrame เป็น Structured Lists พร้อมทำ Data Cleaning เบื้องต้น"""
    import pandas as pd
    if df.empty: return [], []
    try: df = df.map(_clean_thai_text)
    except: df = df.applymap(_clean_thai_text)
    
    # กรองแถวที่ไม่มีข้อมูลที่มีนัยสำคัญ (Empty/Whitespace Rows)
    df_str = df.astype(str)
    mask = df_str.apply(lambda r: any(_has_meaningful_text(c) for c in r), axis=1)
    df = df[mask]
    if df.empty: return [], []
    
    # [Heuristic] ระบุตำแหน่งของหัวตาราง (Header Identification)
    best_idx, best_score = 0, -1
    for i in range(min(3, len(df))):
        score = sum(1 for v in df.iloc[i] if _has_meaningful_text(v))
        if score > best_score: best_score = score; best_idx = i
            
    header = [str(h).strip() for h in df.iloc[best_idx].tolist()]
    rows = [[str(c).strip() for c in row] for _, row in df.iloc[best_idx+1:].iterrows()]
    rows = [r for r in rows if any(_has_meaningful_text(c) for c in r)]
    return header, rows

def _split_rows_by_header(rows: list[list[Any]]) -> list[tuple[str, list[list[Any]]]]:
    """
    [Smart Segmentation] ตรวจหาตารางย่อยที่ซ่อนอยู่ในตารางใหญ่ (เช่น ตารางที่มี Header คั่นกลางแถว)
    เพื่อแยกข้อมูลออกเป็นหมวดหมู่ที่ถูกต้อง (Semantic Splitting)
    """
    blocks = []; curr_head = "Generic Section"; curr_rows = []
    for r in rows:
        rt = " ".join(str(c) for c in r)
        found = next((p for p in HEADER_PATTERNS if re.search(p, rt, re.IGNORECASE)), None)
        if found:
            if curr_rows: blocks.append((curr_head, curr_rows))
            curr_head = _clean_thai_text(found); curr_rows = []
        else:
            cr = [_clean_thai_text(c) for c in r]
            if any(_has_meaningful_text(c) for c in cr): curr_rows.append(cr)
    if curr_rows: blocks.append((curr_head, curr_rows))
    return blocks

def _extract_text_from_html_headers(html: str) -> str:
    """ดึงข้อมูลหัวข้อจาก HTML Table เพื่อใช้ในกระบวนการ Classification"""
    if not html: return ""
    headers = re.findall(r'<th[^>]*>(.*?)</th>', html, re.IGNORECASE | re.DOTALL)
    return " ".join(re.sub(r'<[^>]+>', ' ', h).strip() for h in headers)


# -------------------------------------------------------------------
# Core Entrypoint: extract_tables (The Hybrid Orchestrator)
# -------------------------------------------------------------------

def extract_tables(
    file_path: str | Path,
    doc_id: str,
    doc_type: str = "generic",
    pages: str = "all",
    flavor_priority: Optional[list[str]] = None,
) -> List[TableBlock]:
    """
    กระบวนการหลักในการสกัดข้อมูลตารางแบบไฮบริด (Hybrid Execution Pipeline):
    1. ตรวจสอบพื้นที่ตารางเชิงภาพ (Visual Detection) และสกัดด้วย Vision AI (ถ้ามี API)
    2. สกัดด้วย Camelot (Lattice/Stream) เพื่อความแม่นยำสูงสุดในเชิงโปรแกรม
    3. ทำการ Conflict Resolution: หาก Camelot พบตารางในหน้าเดียวกัน จะให้ความสำคัญกับ Camelot 
       และทิ้งข้อมูลจาก Vision AI เพื่อป้องกัน Data Hallucination หรือการประมวลผลซ้ำซ้อน
    """
    
    path = Path(file_path)
    if not path.exists(): raise FileNotFoundError(f"PDF not found: {path}")

    llm_client = _get_llm_client()
    vision_tables = []; camelot_tables = []
    seen_hashes = set()
    global_ctr = 0

    # --- STRATEGY 1: VISION-BASED EXTRACTION (AI-Powered) ---
    if llm_client:
        try:
            with fitz.open(path) as doc:
                page_indices = range(len(doc))
                # จัดการการประมวลผลแบบระบุหน้า (Page Selection)
                if pages != "all":
                    try:
                        if "-" in pages: s, e = map(int, pages.split("-")); page_indices = range(s-1, e)
                        else: page_indices = [int(p)-1 for p in pages.split(",")]
                    except: pass

                for idx in page_indices:
                    if idx >= len(doc): continue
                    page = doc[idx]
                    
                    # [Heuristic Detection] ตรวจหาพื้นที่ที่มีลายเส้น (Drawings) หนาแน่นซึ่งบ่งบอกถึงตาราง
                    drawings = page.get_drawings()
                    if len(drawings) > 10:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        
                        print(f"[table_extractor] Vision processing page {idx+1}...")
                        html = _extract_table_with_vision(llm_client, img)
                        if not html: continue

                        cols, rows, cx_body, cx_head = parse_html_table(html)
                        if cols and not rows: continue 
                        
                        chash = _compute_row_content_hash(rows)
                        if chash in seen_hashes: continue
                        seen_hashes.add(chash)

                        # Enrichment: สร้างบทสรุปและหมวดหมู่ตารางด้วย AI
                        summary = _summarize_table(llm_client, html, is_html=True)
                        cat = _classify_category_with_llm(llm_client, f"{summary} {_extract_text_from_html_headers(html)}")
                        
                        global_ctr += 1
                        t_id = f"tbl_{doc_id}_{idx+1:03d}_{global_ctr:04d}"
                        
                        # เลือกใช้ HTML format สำหรับตารางที่ซับซ้อน (Complex Merge)
                        md = html if cx_body else table_to_markdown(cols, rows)
                        
                        vision_tables.append(TableBlock(
                            id=t_id, doc_id=doc_id, page=idx+1, name=f"Table {global_ctr} (Vision)",
                            category=cat, columns=cols, rows=rows, markdown=md,
                            bbox=(0,0,0,0),
                            extra={
                                "html_content": html, "summary": summary, "method": "qwen_vision",
                                "source": "vision", "structured_available": not cx_body,
                                "schema_version": SCHEMA_VERSION
                            }
                        ))
        except Exception as e:
            print(f"[table_extractor] Vision Error: {e}")

    # --- STRATEGY 2: PROGRAMMATIC EXTRACTION (Camelot) ---
    print("[table_extractor] Using Camelot...")
    if flavor_priority is None: flavor_priority = ["lattice", "stream"]

    for flavor in flavor_priority:
        try:
            tables = camelot.read_pdf(str(path), pages=pages, flavor=flavor)
        except: continue
        if tables.n == 0: continue

        for t in tables:
            cols, rows = _dataframe_to_columns_rows(t.df)
            if len(cols) < MIN_COLS: continue
            
            # ทำการแบ่งตารางย่อยหากพบแพทเทิร์นหัวข้อใหม่ (Smart Table Splitting)
            sub_tables = _split_rows_by_header(rows)
            items = sub_tables if sub_tables else [("Table", rows)]

            for head_txt, sub_rows in items:
                if len(sub_rows) <= 1: continue 
                
                chash = _compute_row_content_hash(sub_rows)
                if chash in seen_hashes: continue
                seen_hashes.add(chash)

                global_ctr += 1
                md = table_to_markdown(cols, sub_rows)
                html = table_to_html(cols, sub_rows)
                
                # AI Enrichment Layer
                summary = _summarize_table(llm_client, md, is_html=False)
                cat = _classify_category_with_llm(llm_client, f"{head_txt} {' '.join(cols)}")
                
                t_id = f"tbl_{doc_id}_{t.page:03d}_{global_ctr:04d}"

                camelot_tables.append(TableBlock(
                    id=t_id, doc_id=doc_id, page=t.page, name=head_txt,
                    category=cat, columns=cols, rows=sub_rows, markdown=md,
                    bbox=None,
                    extra={
                        "html_content": html, "summary": summary, "method": "camelot",
                        "source": "camelot", "structured_available": True,
                        "numeric_trust": "high", "schema_version": SCHEMA_VERSION
                    }
                ))

    # --- STRATEGY MERGE & CONFLICT RESOLUTION ---
    # ลอจิกการผสานข้อมูล: หากหน้าใดที่มีข้อมูลจาก Camelot (High Trust) เราจะทิ้งข้อมูลจาก Vision AI ทันที
    final_tables = []
    camelot_pages = set(t.page for t in camelot_tables)
    final_tables.extend(camelot_tables)
    
    for vt in vision_tables:
        if vt.page not in camelot_pages:
            final_tables.append(vt)
        else:
            print(f"[table_extractor] Dropping Vision table on page {vt.page} (Camelot found).")

    final_tables.sort(key=lambda x: (x.page, x.id))
    print(f"[table_extractor] Extracted {len(final_tables)} tables.")
    return final_tables


# -------------------------------------------------------------------
# Data Connectivity: Interface for Other Modules
# -------------------------------------------------------------------

def table_to_text(table: TableBlock) -> str:
    """
    [Data Textualization] แปลง TableBlock เป็นข้อความสำหรับการทำ Vector Embedding:
    - รวบรวมข้อมูลชื่อตาราง, บทสรุป และ Markdown Content
    - ช่วยให้ตารางถูกค้นหาเจอผ่านระบบ Semantic Search (RAG)
    """
    parts = []
    if table.name:
        parts.append(f"Table: {table.name}")
    
    summary = table.extra.get("summary")
    if summary:
        parts.append(f"Summary: {summary}")
        
    if table.markdown:
        parts.append(table.markdown)
    else:
        # Fallback Strategy: นำข้อมูลคอลัมน์และแถวมาต่อกันเป็นข้อความเรียบง่าย
        if table.columns:
            parts.append(" | ".join(map(str, table.columns)))
        for row in table.rows[:10]: # จำกัดจำนวนแถวเพื่อรักษา Token efficiency
            parts.append(" | ".join(map(str, row)))
            
    return "\n".join(parts)


def compute_from_table(table: TableBlock, operation: str = "sum", column: str = None):
    """
    [Arithmetic Engine] ทำการคำนวณตัวเลขในตาราง (รองรับเฉพาะ High Trust Tables):
    - ป้องกันการคำนวณผิดพลาดจากการสกัดข้อมูลด้วย Vision (Security/Safety Gate)
    - รองรับ Operation พื้นฐาน: Sum, Mean, Max
    """
    if table.extra.get("numeric_trust") != "high":
        # ไม่อนุญาตให้คำนวณข้อมูลจากตาราง Vision เนื่องจากความเสี่ยงเรื่อง AI Hallucination
        raise ValueError("Low trust table (Vision-based). Calculation unsafe.")
        
    import pandas as pd
    try: df = pd.DataFrame(table.rows, columns=table.columns)
    except: raise ValueError("DF creation failed")
    
    # พยายามระบุคอลัมน์ที่เป็นตัวเลขโดยอัตโนมัติหากไม่ได้ระบุมา
    if column is None:
        for c in df.columns:
            try: df[c] = pd.to_numeric(df[c]); column = c; break
            except: pass
            
    if not column: raise ValueError("No numeric column")
    
    if operation == "sum": return df[column].sum()
    elif operation == "mean": return df[column].mean()
    elif operation == "max": return df[column].max()
    return 0