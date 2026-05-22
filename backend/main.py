from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
import asyncio
import shutil
import subprocess
import sys
import os
import re
import time
import hashlib
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal services
from .services.logger import append_log, read_logs
from .services.rag import answer_question
from .services.vector_store import reset_vector_store_cache
from .services.query_router import route_query
from .services.sql_agent import generate_and_run_sql

# -----------------------------------------------------------
# กำหนด Path สำหรับเก็บข้อมูลระบบ
# -----------------------------------------------------------
INGESTED_DIR = Path("ingested")     # โฟลเดอร์เก็บข้อมูลที่ผ่านการสกัดแล้ว (JSON, รูปภาพ)
CHROMA_DB_DIR = Path("chroma_db")   # โฟลเดอร์เก็บฐานข้อมูลเวกเตอร์
UPLOAD_DIR = Path("uploads")        # โฟลเดอร์เก็บไฟล์ PDF ต้นฉบับที่ผู้ใช้อัปโหลดมา

# -----------------------------------------------------------
# ตั้งค่า FastAPI Application
# -----------------------------------------------------------
app = FastAPI(
    title="AI Data Ingestion Backend",
    description="Backend for DB, Embeddings, RAG, API, and Evaluation",
    version="0.2.2 (Multi-Doc Final)",
)

# CORS Middleware: อนุญาตให้คนในเครือข่ายมหาวิทยาลัยเข้าถึงได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตทุก origin ในเครือข่ายท้องถิ่น
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount Frontend: ให้บริการไฟล์ Static สำหรับหน้าเว็บ UI
frontend_path = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/app", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

# 2. Mount Ingested Data: เปิดให้ Frontend เข้าถึงไฟล์รูปภาพและข้อมูลที่สกัดแล้ว
INGESTED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/ingested", StaticFiles(directory=str(INGESTED_DIR)), name="ingested")

# 3. ตรวจสอบและสร้างโฟลเดอร์ Upload
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# Helper: ฟังก์ชันแปลงชื่อไฟล์เพื่อป้องกันปัญหา Path ภาษาไทย
# -----------------------------------------------------------
def _normalize_id(raw_id: str) -> str:
    """
    แปลงรหัสเอกสาร (doc_id) ให้เป็น MD5 Hash เพื่อป้องกันปัญหา 
    URL หรือ Path พังเมื่อใช้ภาษาไทยหรืออักขระพิเศษ
    """
    if not raw_id:
        return "unknown_doc"
    
    hashed = hashlib.md5(raw_id.encode('utf-8')).hexdigest()
    return f"doc_{hashed[:12]}" 

# -----------------------------------------------------------
# API: ตรวจสอบสถานะการทำงานของเซิร์ฟเวอร์ (Health Check)
# -----------------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "backend",
        "mode": "multi_doc",
        "features": ["hybrid_ingestion", "ocr", "rag"],
    }


# -----------------------------------------------------------
# API: ระบบถาม-ตอบด้วย AI (RAG + Hybrid Rendering)
# -----------------------------------------------------------
class AskRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None
    top_k: int = 20
    mode: Literal["auto", "text", "table", "both"] = "auto"

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]
    intent: str
    mode: str
    tables: List[Dict[str, Any]] = []

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    # บันทึก Log เมื่อเซิร์ฟเวอร์ได้รับคำถามจากผู้ใช้
    print(f"👉 [API] ได้รับคำถามแล้ว: '{req.query}' | กำลังส่งให้ AI ประมวลผล...", flush=True)

    # 1. ตรวจสอบและแปลงรูปแบบ ID เอกสารให้อยู่ในรูปแบบที่ถูกต้อง (doc_xxxx)
    sanitized_doc_ids = None
    if req.doc_ids:
        sanitized_doc_ids = []
        for did in req.doc_ids:
            if not did: 
                continue
            # หาก ID ถูกแปลงรหัสมาแล้ว (ขึ้นต้นด้วย doc_) ให้ใช้งานได้เลย
            if did.startswith("doc_"):
                sanitized_doc_ids.append(did)
            else:
                sanitized_doc_ids.append(_normalize_id(did))

    # 2. ตัดสินใจเส้นทาง: SQL Agent หรือ RAG Pipeline
    route = await asyncio.to_thread(route_query, req.query)
    print(f"🔀 [API] Query routed to: [{route.upper()}]", flush=True)

    if route == "sql":
        # เส้นทาง SQL: ดึงข้อมูลโครงสร้างจาก MySQL Database โดยตรง
        sql_answer = await asyncio.to_thread(generate_and_run_sql, req.query)
        result = {
            "answer": sql_answer,
            "sources": [],
            "intent": "sql",
            "mode": "sql",
            "tables": [],
        }
    else:
        # เส้นทาง RAG: ค้นหาจาก Vector Store (เอกสาร PDF / ข้อมูลเชิงบรรยาย)
        result = await answer_question(
            query=req.query,
            doc_ids=sanitized_doc_ids,
            top_k=req.top_k,
            mode=req.mode,
        )

    # บันทึก Log เมื่อระบบสร้างคำตอบเสร็จสิ้น
    print(f"✅ [API] AI ประมวลผลคำตอบเสร็จสิ้น! กำลังเตรียมข้อมูลสำหรับแสดงผล...", flush=True)

    # 3. Post-Processing: ตกแต่งคำตอบโดยการแทนที่แท็ก [SHOW_TABLE] ด้วย HTML Tag ของจริง
    answer_text = result.get("answer", "")
    sources = result.get("sources", [])
    
    table_tags = re.findall(r"\[SHOW_TABLE:CAT=(.*?)\]", answer_text)

    for category_key in table_tags:
        clean_cat = category_key.strip()
        replacement_html = ""

        # ค้นหาแหล่งที่มาที่ตรงกับแท็กเพื่อดึงข้อมูลรูปภาพหรือ HTML ตาราง
        for src in sources:
            metadata = src.get("metadata", src)
            
            is_table_source = src.get("source") == "table" or metadata.get("source") == "table"
            is_image_source = src.get("source") == "image" or metadata.get("source") == "image"
            
            if is_table_source or is_image_source:
                src_cat = metadata.get("category", "")
                if (src_cat == clean_cat) or (clean_cat == ""):
                    
                    # กรณีที่ 1: แหล่งข้อมูลเป็นรูปภาพ (ตารางที่ซับซ้อน)
                    image_path = metadata.get("image_path") or metadata.get("extra", {}).get("image_path")
                    if image_path:
                        doc_id = metadata.get("doc_id")
                        full_img_url = f"/ingested/{doc_id}/{image_path}"
                        replacement_html = (
                            f"<div class='my-4 p-2 border rounded bg-slate-50 text-center'>"
                            f"<p class='text-xs text-slate-500 mb-1'>Original Form (Complex Layout)</p>"
                            f"<img src='{full_img_url}' alt='Table Image' "
                            f"class='max-w-full h-auto rounded shadow-sm mx-auto border' />"
                            f"</div>"
                        )
                        break

                    # กรณีที่ 2: แหล่งข้อมูลเป็นตาราง HTML
                    html_content = metadata.get("html_content") or metadata.get("extra", {}).get("html_content")
                    if html_content:
                        replacement_html = f"<br><div class='table-responsive answer-tables-content'>{html_content}</div><br>"
                        break
        
        # แทนที่รหัสแท็กด้วย HTML ที่พร้อมแสดงผลบนหน้าเว็บ
        tag_str = f"[SHOW_TABLE:CAT={category_key}]"
        if replacement_html:
            answer_text = answer_text.replace(tag_str, replacement_html)
        else:
            answer_text = answer_text.replace(tag_str, "")

    # ลบ tag [SHOW_TABLE:TBL_xxx] ที่ยังค้างอยู่ (hallucinated tags จาก LLM ที่ไม่ถูก resolve)
    answer_text = re.sub(r"\[SHOW_TABLE:[^\]]+\]", "", answer_text).strip()

    result["answer"] = answer_text

    # 4. บันทึกประวัติการสนทนาลงในระบบ Log
    try:
        append_log({
            "query": req.query, "doc_ids": req.doc_ids,
            "answer": result.get("answer"), "intent": result.get("intent")
        })
    except Exception as e:
        print(f"[LOG_ERROR] {e!r}")

    result["tables"] = result.get("tables", [])
    return AskResponse(**result)


# -----------------------------------------------------------
# API: ดึงประวัติการสนทนาย้อนหลัง
# -----------------------------------------------------------
class HistoryItem(BaseModel):
    ts: str
    query: str
    answer: str
    doc_ids: Optional[List[str]] = None
    intent: Optional[str] = None
    mode: Optional[str] = None

@app.get("/history", response_model=List[HistoryItem])
def get_history(limit: int = 50):
    logs = read_logs(limit=limit)
    items = []
    for e in logs:
        items.append(HistoryItem(
            ts=e.get("ts", ""), query=e.get("query", ""), answer=e.get("answer", ""),
            doc_ids=e.get("doc_ids"), intent=e.get("intent"), mode=e.get("mode")
        ))
    return items


# -----------------------------------------------------------
# API: อัปโหลดและประมวลผลไฟล์ PDF (Multi-Document Mode)
# -----------------------------------------------------------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    doc_type: str = Form(""),
    use_ocr: bool = Form(True),
):
    # 0. กำหนดค่าเริ่มต้นหากไม่ได้ระบุประเภทเอกสาร
    if not doc_type.strip(): doc_type = "generic_doc"

    # 1. ตรวจสอบความถูกต้องของไฟล์นามสกุลและข้อมูลนำเข้า
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ PDF เท่านั้น")
    if not doc_id.strip():
        raise HTTPException(status_code=400, detail="ต้องระบุ doc_id")

    safe_doc_id = _normalize_id(doc_id)
    print(f"[UPLOAD] Received doc_id='{doc_id}' -> normalized='{safe_doc_id}'")

    # 2. ยืนยันว่าโฟลเดอร์สำหรับเก็บไฟล์มีอยู่จริง
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    # 3. บันทึกไฟล์ PDF ลงในระบบ
    dest_path = UPLOAD_DIR / f"{safe_doc_id}.pdf"
    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    # จัดเก็บชื่อไฟล์ภาษาไทยต้นฉบับไว้ใน meta.json เพื่อนำไปแสดงผลภายหลัง
    doc_ingest_dir = INGESTED_DIR / safe_doc_id
    doc_ingest_dir.mkdir(parents=True, exist_ok=True)
    with open(doc_ingest_dir / "meta.json", "w", encoding="utf-8") as meta_f:
        json.dump({"original_name": doc_id}, meta_f, ensure_ascii=False)

    # 4. เรียกใช้สคริปต์สกัดข้อมูลจาก PDF (Parsing & Enrichment)
    try:
        print(f"[UPLOAD] 🛑 Releasing DB lock before ingestion...")
        reset_vector_store_cache()

        script_name = "scripts.run_ingestion" if use_ocr else "scripts.run_all"
        cmd = [
            sys.executable, "-m", script_name,
            str(dest_path),
            "--doc-id", safe_doc_id,
            "--doc-type", doc_type,
            "--output-root", str(INGESTED_DIR) 
        ]
        if script_name == "scripts.run_ingestion" and not use_ocr:
            cmd.append("--no-ocr")
            
        print(f"[UPLOAD] Running pipeline: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {e}")

    # 5. นำข้อมูลที่สกัดได้เข้าสู่ Vector Database (ChromaDB)
    reset_vector_store_cache()
    try:
        # สคริปต์ ingest_doc จะสแกนไฟล์ที่ถูกประมวลผลแล้วและบันทึกลง Database
        cmd = [sys.executable, "-m", "scripts.ingest_doc"]
        print(f"[UPLOAD] Re-indexing (All Docs): {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print("[UPLOAD] ⏳ Waiting for DB lock release (3s)...")
        time.sleep(3)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Re-index failed: {e}")

    # เคลียร์แคชระบบฐานข้อมูลหลังอัปเดตเสร็จ
    reset_vector_store_cache()

    return {
        "ok": True,
        "doc_id": safe_doc_id,
        "original_doc_id": doc_id,
        "doc_type": doc_type,
        "message": "File uploaded and ingested successfully (Append Mode).",
        "pipeline": "hybrid_ingestion",
    }

# -----------------------------------------------------------
# API: แสดงรายการเอกสารทั้งหมดในระบบ (List Documents)
# -----------------------------------------------------------
@app.get("/documents")
def list_documents():
    docs = []
    if INGESTED_DIR.exists():
        for item in INGESTED_DIR.iterdir():
            if item.is_dir():
                doc_name = item.name # กำหนดค่าเริ่มต้นเป็นรหัส Hash
                
                # พยายามดึงชื่อเอกสารต้นฉบับ (ภาษาไทย) จากไฟล์ meta.json
                meta_file = item / "meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as meta_f:
                            meta_data = json.load(meta_f)
                            doc_name = meta_data.get("original_name", doc_name)
                    except Exception:
                        pass

                docs.append({
                    "id": item.name,   # ใช้ ID รูปแบบ Hash สำหรับประมวลผลหลังบ้าน
                    "name": doc_name   # ใช้ชื่อต้นฉบับสำหรับแสดงผลบนหน้าจอ UI
                })
    
    # เรียงลำดับเอกสารตามตัวอักษร
    docs.sort(key=lambda x: x["name"])
    return {"documents": docs}

# -----------------------------------------------------------
# API: ตรวจสอบสถานะและข้อมูลภายใน ChromaDB (Database Dashboard)
# -----------------------------------------------------------
from .services.vector_store import get_collection_info, get_vector_store

@app.get("/api/database/stats")
def get_db_stats():
    """ดึงสถิติภาพรวมของฐานข้อมูลเวกเตอร์"""
    info = get_collection_info()
    if "error" in info:
        return {"status": "error", "message": info["error"]}
    return {"status": "success", "data": info}

@app.get("/api/database/sample")
def get_db_samples(limit: int = 10):
    """สุ่มตัวอย่างข้อมูลดิบจาก Vector Database"""
    try:
        vectordb = get_vector_store()
        collection = vectordb._collection
        raw_data = collection.get(limit=limit)
        
        results = []
        if raw_data and raw_data.get('documents'):
            for i in range(len(raw_data['documents'])):
                results.append({
                    "id": raw_data['ids'][i],
                    "metadata": raw_data['metadatas'][i],
                    "text": raw_data['documents'][i]
                })
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -----------------------------------------------------------
# API: Redirect หน้าแรกไปยังเว็บแอปพลิเคชัน
# -----------------------------------------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/app/index.html")