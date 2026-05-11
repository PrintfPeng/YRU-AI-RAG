# ingestion/docling_parser.py

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

# Docling: Advanced Document Parsing Framework
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    EasyOcrOptions, # การเลือกใช้ EasyOCR เพื่อลด Complexity ในการติดตั้งบน Windows Environment
)

# Project-specific Schemas
from .schema import (
    IngestedDocument,
    TableBlock,
    TextBlock,
    DocumentMetadata,
    ImageBlock,
)


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# MAIN PARSER (Primary Table & Text Extractor)
# -------------------------------------------------------------------
class DoclingParser:
    """
    Parser หลักที่ใช้ Docling Engine ในการแปลงไฟล์ PDF เป็น Structured Data
    รองรับการทำ OCR ภาษาไทย/อังกฤษ และการวิเคราะห์โครงสร้างตารางระดับสูง
    """
    def __init__(self, config=None):
        self.config = config

        # กำหนดค่า Pipeline สำหรับการประมวลผล PDF
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        
        # [Technical Decision] ใช้ EasyOCR เป็น Engine หลักเพื่อรองรับ Multi-language (TH/EN) 
        # และลดปัญหาเรื่อง Dependency บนระบบปฏิบัติการ Windows
        pipeline_options.ocr_options = EasyOcrOptions(lang=["th", "en"])
        
        # ตั้งค่าการวิเคราะห์โครงสร้างตารางและการจับคู่ Cell ข้อมูล
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )
        
        # ตั้งค่าการเรนเดอร์รูปภาพหน้ากระดาษเพื่อใช้ในการทำ Visual Debugging หรือ Manual Review
        pipeline_options.generate_page_images = True
        pipeline_options.images_scale = 3.0
        self.image_scale = 3.0

        # Initialize Converter พร้อมกำหนดรูปแบบการประมวลผล PDF ตาม Options ที่ตั้งไว้
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse(self, file_path: str) -> IngestedDocument:
        """
        Entry point สำหรับการ Parsing ไฟล์ PDF:
        - แปลงเอกสารเป็นรหัสภายใน (Internal representation)
        - ประมวลผลรูปภาพประกอบหน้ากระดาษ (Page Images)
        - สกัด Text และ Table blocks พร้อม Metadata
        """
        logger.info(f"Starting Docling parse for: {file_path}")
        try:
            conv_res = self.converter.convert(file_path)
            doc = conv_res.document

            # ดึงข้อมูลรูปภาพหน้ากระดาษ (Page Rendering) สำหรับใช้ในกระบวนการถัดไป
            page_images = {}
            for page_no, page in doc.pages.items():
                img = None
                # ตรวจสอบหา PIL Image จาก Property ต่างๆ ของ Docling Page Object
                if hasattr(page, "image") and page.image:
                    if hasattr(page.image, "pil_image"):
                        img = page.image.pil_image
                    elif hasattr(page.image, "image"):
                        img = page.image.image
                    else:
                        img = page.image

                # Fallback Logic: หากไม่พบรูปภาพใน Property ให้ทำการ Force Render ใหม่ที่ Scale 3.0
                if img is None:
                    try:
                        img = page.get_image(scale=3.0)
                    except Exception as e:
                        logger.warning(
                            f"Could not render image for page {page_no}: {e}"
                        )

                if img:
                    page_images[page_no] = img

            doc_id = Path(file_path).stem

            # สกัดเนื้อหาประเภทข้อความ (Plain Text)
            text_blocks = self._process_text(doc, doc_id)

            # กำหนด Output Directory สำหรับจัดเก็บ Artifacts (เช่น รูปตารางที่ตัดออกมา)
            output_dir = None
            if self.config and getattr(self.config, "output_dir", None):
                output_dir = self.config.output_dir
            else:
                output_dir = str(Path("ingested") / doc_id)

            # ประมวลผลตารางและสกัดรูปภาพตาราง (Table Extraction & Cropping)
            table_blocks, table_images = self._process_tables(
                doc, page_images, doc_id, output_dir
            )

            # รวบรวมข้อมูล Metadata ของเอกสารสำหรับการทำ Indexing
            metadata = DocumentMetadata(
                doc_id=doc_id,
                file_name=Path(file_path).name,
                doc_type="generic",
                page_count=len(doc.pages),
                ingested_at=datetime.now().isoformat(),
                source="docling",
            )

            return IngestedDocument(
                metadata=metadata,
                texts=text_blocks,
                tables=table_blocks,
                images=table_images,
            )

        except Exception as e:
            logger.error(f"Docling parsing failed: {e}")
            raise

    def _process_text(self, doc, doc_id: str) -> List[TextBlock]:
        """วนลูปสกัด TextItem จากโครงสร้างเอกสารและแปลงเป็น TextBlock Schema"""
        blocks = []
        for i, (item, level) in enumerate(doc.iterate_items()):
            if item.__class__.__name__ == "TextItem":
                # กรองเนื้อหาที่เป็นค่าว่างหรือ Whitespace ล้วนออก
                if not item.text.strip():
                    continue
                
                # ตรวจสอบตำแหน่ง (Provenance) ของข้อความในหน้าเอกสาร
                page_no = item.prov[0].page_no if item.prov else 1
                bbox = item.prov[0].bbox.as_tuple() if item.prov else None
                
                blocks.append(
                    TextBlock(
                        id=f"text_{i}",
                        doc_id=doc_id,
                        content=item.text,
                        page=page_no,
                        bbox=bbox,
                        extra={"role": "paragraph"},
                    )
                )
        return blocks

    # -------------------------------------------------------------------
    # OpenCV Table Detection (Experimental/Auxiliary)
    # -------------------------------------------------------------------
    def _detect_table_regions(self, pil_image):
        """
        ใช้ Computer Vision (OpenCV) ในการวิเคราะห์หาพื้นที่ที่น่าจะเป็นตารางบนหน้ากระดาษ
        - ใช้ Morphological operations เพื่อตรวจหาเส้นตารางแนวตั้งและแนวนอน
        - คืนค่าเป็นพิกัดล้อมรอบ (Bounding Boxes) ของตารางที่พบ
        """
        img = np.array(pil_image)

        # การประมวลผลภาพเบื้องต้น (Pre-processing)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray) # ปรับความคมชัดเพื่อให้เส้นตารางชัดเจนขึ้น

        # ทำ Adaptive Thresholding เพื่อแยกเส้นออกจากพื้นหลัง
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            5,
        )

        h, w = gray.shape

        # คำนวณ Dynamic Kernel Size ตามขนาดของรูปภาพหน้ากระดาษ
        horizontal_len = max(40, w // 25)
        vertical_len = max(40, h // 25)

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (horizontal_len, 1)
        )
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_len))

        # สกัดเส้นแนวตั้งและแนวนอน (Morphological Opening)
        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

        # รวมเส้นแนวตั้งและแนวนอนเพื่อสร้าง Mask ของตาราง
        table_mask = cv2.add(horizontal, vertical)

        # ขยายขอบเขตของ Mask เพื่อเชื่อมรอยต่อที่ขาดหาย (Dilation)
        kernel = np.ones((5, 5), np.uint8)
        table_mask = cv2.dilate(table_mask, kernel, iterations=2)

        # ค้นหาเส้นขอบ (Contours) จาก Mask ที่สร้างขึ้น
        contours, _ = cv2.findContours(
            table_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch

            # กรองเฉพาะพื้นที่ที่มีขนาดและความกว้าง/สูง ตามเกณฑ์ขั้นต่ำของตารางมาตรฐาน
            if cw > 200 and ch > 80 and area > (w * h * 0.01):
                regions.append((x, y, x + cw, y + ch))

        return regions

    def _match_best_region(self, regions, docling_bbox):
        """
        เปรียบเทียบพิกัดตารางที่ Docling ตรวจพบ กับพิกัดที่ OpenCV วิเคราะห์ได้ (Intersection over Union)
        เพื่อเลือกพื้นที่ที่ครอบคลุมเนื้อหาตารางได้สมบูรณ์ที่สุด
        """
        if not regions:
            return None

        if not docling_bbox:
            return regions[0]

        dx0, dy0, dx1, dy1 = docling_bbox

        best_score = 0
        best_region = None

        for x0, y0, x1, y1 in regions:
            # คำนวณพื้นที่ทับซ้อน (Intersection Area)
            inter_x0 = max(x0, dx0)
            inter_y0 = max(y0, dy0)
            inter_x1 = min(x1, dx1)
            inter_y1 = min(y1, dy1)

            if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                continue

            inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
            box_area = (x1 - x0) * (y1 - y0)

            # คำนวณคะแนนความเชื่อมั่น (Confidence Score)
            score = inter_area / float(box_area)

            if score > best_score:
                best_score = score
                best_region = (x0, y0, x1, y1)

        return best_region

    def _process_tables(
        self,
        doc,
        page_images: dict,
        doc_id: str,
        output_dir: str,
    ) -> Tuple[List[TableBlock], List[ImageBlock]]:
        """
        ประมวลผลตารางในเอกสาร:
        - แปลงข้อมูลเป็น DataFrame และ Markdown
        - ดึงรูปภาพตาราง (Native Cropping) และจัดเก็บลง Disk
        - สร้าง ImageBlock และ TableBlock สำหรับใช้งานในขั้นต่อไป
        """
        blocks = []
        img_blocks = []

        # เตรียม Directory สำหรับจัดเก็บไฟล์รูปภาพตารางที่สกัดได้
        img_output_dir = os.path.join(output_dir, "images")
        os.makedirs(img_output_dir, exist_ok=True)

        for i, table in enumerate(doc.tables):
            # สกัดข้อมูลตารางในรูปแบบโครงสร้าง (DataFrame) และกึ่งโครงสร้าง (Markdown)
            df = table.export_to_dataframe(doc)
            md = table.export_to_markdown(doc)

            saved_image_path = None
            page_no = table.prov[0].page_no if table.prov else 1
            bbox_tuple = table.prov[0].bbox.as_tuple() if table.prov else None

            # [Enrichment] สร้าง Caption โดยใช้ Markdown Preview เพื่อให้ AI เข้าใจเนื้อหาในตารางภาพ
            table_content_preview = md[:300].replace("\n", " ")
            table_caption = f"Table {i+1}: {table_content_preview}..."

            # [Optimization] ใช้ Native Engine ของ Docling ในการ Crop รูปตารางโดยตรง 
            # เพื่อความแม่นยำสูงสุดและลด Overhead จากการคำนวณพิกัดเอง
            try:
                img_obj = table.get_image(doc)
                if img_obj:
                    filename = f"table_p{page_no:03d}_{i:03d}.png"
                    full_path = os.path.join(img_output_dir, filename)

                    # บันทึกไฟล์รูปภาพลงในโฟลเดอร์ผลลัพธ์
                    img_obj.save(full_path, "PNG")
                    saved_image_path = f"images/{filename}"

                    # สร้าง ImageBlock สำหรับเป็น Metadata ของรูปภาพตาราง
                    img_blocks.append(
                        ImageBlock(
                            id=f"img_tbl_{i}",
                            doc_id=doc_id,
                            page=page_no,
                            file_path=saved_image_path,
                            caption=table_caption,
                            bbox=bbox_tuple,
                            section="table",
                            category="table_image",
                            extra={
                                "source": "docling_native",
                                "markdown": md,
                            },
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to extract table image natively: {e}")

            # สร้าง TableBlock ที่รวบรวมทั้งข้อมูลโครงสร้างและลิงก์ไปยังไฟล์รูปภาพ
            blocks.append(
                TableBlock(
                    id=f"TBL_{i}",
                    doc_id=doc_id,
                    page=page_no,
                    columns=[str(c) for c in df.columns] if df is not None else [],
                    rows=df.values.tolist() if df is not None else [],
                    markdown=md,
                    image_path=saved_image_path,
                    is_complex=True,
                    source="docling",
                    structured_available=bool(df is not None and not df.empty),
                    bbox=bbox_tuple,
                )
            )

        return blocks, img_blocks


# -------------------------------------------------------------------
# IMAGE PARSER (General Non-table Image Extractor)
# -------------------------------------------------------------------
class DoclingImageParser:
    """
    Parser แยกส่วนสำหรับสกัดรูปภาพประเภทอื่นๆ (Pictures/Figures) ที่ไม่ใช่ตารางออกจากเอกสาร
    โดยเน้นการใช้ Picture Extraction Engine ของ Docling
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # ตั้งค่า Pipeline เฉพาะสำหรับการสกัดรูปภาพเท่านั้น
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract_images(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        สแกนและบันทึกรูปภาพทั้งหมดที่พบในเอกสาร พร้อมทั้งจัดการ Path ให้เป็นมิตรกับ Web Browser
        """
        file_path = Path(pdf_path).resolve()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"[DoclingImageParser] Scanning images in: {file_path.name} ...")
        try:
            conv_res = self.converter.convert(file_path)
            doc = conv_res.document
            saved_images = []
            
            if hasattr(doc, "pictures") and doc.pictures:
                print(f"[DoclingImageParser] Found {len(doc.pictures)} images.")
                for i, picture in enumerate(doc.pictures):
                    page_no = 1
                    bbox = None
                    if picture.prov and picture.prov[0]:
                        page_no = picture.prov[0].page_no
                        if hasattr(picture.prov[0], "bbox"):
                            bbox = picture.prov[0].bbox.as_tuple()

                    image_filename = f"img_p{page_no:03d}_{i+1:03d}.png"
                    image_save_path = out_path / image_filename

                    # บันทึกรูปภาพโดยใช้ Native rendering ของ Docling
                    img_obj = picture.get_image(doc)
                    if img_obj:
                        img_obj.save(image_save_path, "PNG")

                        # [Normalization] แปลง Absolute Path ให้เป็น Relative Path ที่เรียกใช้ได้ผ่าน Web
                        try:
                            # คำนวณ Relative Path จาก Current Working Directory
                            rel_path = os.path.relpath(image_save_path, os.getcwd())
                            # มาตรฐาน Posix (/) สำหรับการใช้งานบน Web Browser และ Cross-platform compatibility
                            rel_path = rel_path.replace("\\", "/")
                        except ValueError:
                            # กรณีไฟล์อยู่คนละ Drive (เช่น บน Windows) จะคืนค่าเป็น Absolute String
                            rel_path = str(image_save_path)

                        saved_images.append(
                            {
                                "index": i,
                                "file_path": rel_path,
                                "file_name": image_filename,
                                "page": page_no,
                                "bbox": bbox,
                            }
                        )
            return saved_images
        except Exception as e:
            print(f"❌ [DoclingImageParser] Error: {e}")
            return []