# backend/services/vector_store.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
import warnings
import re
import gc  # นำเข้าเพื่อใช้จัดการกระบวนการคืนหน่วยความจำและปลดล็อกไฟล์ (Memory/File Lock Management)

from fastapi import HTTPException
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .chunking import Chunk
from .embeddings import get_embedding_model


# -----------------------------------------------------------
# System Setup & Error Handling
# -----------------------------------------------------------
logger = logging.getLogger(__name__)

# ดักจับข้อผิดพลาดเชิงลึกของฐานข้อมูล ChromaDB เพื่อนำไปทำระบบประมวลผลสำรอง (Auto-healing)
try:
    from chromadb.errors import InternalError as ChromaInternalError
except Exception:
    ChromaInternalError = Exception

# -----------------------------------------------------------
# Database Configuration
# -----------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "yru_planning_data"

# ระบบแคชหน่วยความจำ (In-memory Cache) สำหรับลดภาระการเชื่อมต่อฐานข้อมูลซ้ำซ้อน
_vectordb_cache: Dict[Tuple[str, str], Chroma] = {}


# -----------------------------------------------------------
# Helper: Document ID Sanitization
# -----------------------------------------------------------
def sanitize_doc_id(doc_id: str) -> str:
    """
    จัดระเบียบและทำความสะอาดรหัสเอกสาร (Document ID) ให้สอดคล้องกับมาตรฐานของระบบฐานข้อมูล
    รองรับอักขระภาษาไทยและป้องกันข้อผิดพลาดจากการอ้างอิง ID ที่มีอักขระพิเศษ
    """
    if not doc_id:
        return ""
    doc_id = doc_id.lower().strip()
    doc_id = re.sub(r'\s+', '_', doc_id)
    doc_id = re.sub(r'[^a-z0-9_\u0E00-\u0E7F-]', '', doc_id) 
    return doc_id


# -----------------------------------------------------------
# Core: Chroma Database Management
# -----------------------------------------------------------
def _cache_key(persist_directory: str, collection_name: str) -> Tuple[str, str]:
    """สร้างคีย์อ้างอิงสำหรับระบบแคชของฐานข้อมูล"""
    return (str(Path(persist_directory).resolve()), collection_name)


def get_vector_store(
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    force_recreate: bool = False,
    reload: bool = False, 
) -> Chroma:
    """
    สร้างและคืนค่าอินสแตนซ์ของ Chroma Vector Store พร้อมเชื่อมต่อกับ Embedding Client
    รองรับระบบแคชภายใน (In-process Cache) และฟังก์ชันการบังคับรีโหลด (Force Reload)
    """
    should_reload = force_recreate or reload 

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    key = _cache_key(persist_directory, collection_name)

    # จัดการกระบวนการรีโหลดฐานข้อมูล: ลบแคชและบังคับปลดล็อกไฟล์ผ่าน Garbage Collector
    if should_reload and key in _vectordb_cache:
        logger.info(f"[vector_store] Forcing reload of ChromaDB client for {key}")
        del _vectordb_cache[key]
        gc.collect() 

    if not force_recreate and key in _vectordb_cache:
        return _vectordb_cache[key]

    embeddings = get_embedding_model()
    
    import os
    import chromadb
    chroma_host = os.getenv("CHROMA_SERVER_HOST")
    chroma_port = os.getenv("CHROMA_SERVER_PORT", "8000")

    try:
        if chroma_host:
            logger.info(f"[vector_store] Connecting to remote ChromaDB at {chroma_host}:{chroma_port}")
            client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            vectordb = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vectordb = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=str(persist_path),
                )
    except Exception as e:
        logger.exception("[vector_store] Failed to init Chroma: %s", e)
        # กลไกการฟื้นฟูระบบ (Auto-recovery): บังคับรวบรวมขยะในหน่วยความจำและทดลองเชื่อมต่ออีกครั้ง
        gc.collect()
        try:
            if chroma_host:
                client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
                vectordb = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=embeddings,
                )
            else:
                vectordb = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=str(persist_path),
                )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="ไม่สามารถเชื่อมต่อ Vector DB ได้ โปรดตรวจสอบการติดตั้ง"
            ) from e

    _vectordb_cache[key] = vectordb
    return vectordb

# -----------------------------------------------------------------------------
# Core: Global Cache Reset (Memory & File Lock Management)
# -----------------------------------------------------------------------------
def reset_vector_store_cache():
    """
    บังคับล้างแคชฐานข้อมูลแบบสมบูรณ์แบบ (Global Reset) 
    มักใช้หลังกระบวนการนำเข้าข้อมูล (Data Ingestion) เพื่อให้ระบบโหลดดัชนี (Index) ล่าสุดจากดิสก์
    พร้อมแก้ปัญหาไฟล์ถูกล็อก (File Lock) ในระบบปฏิบัติการ Windows
    """
    global _vectordb_cache
    
    if _vectordb_cache:
        print(f"[vector_store] 🧹 Force clearing cache: {len(_vectordb_cache)} entries...")
        _vectordb_cache.clear()
    else:
        print("[vector_store] 🧹 Cache is already empty.")
    
    # บังคับรวบรวมขยะหน่วยความจำ (Garbage Collection) เพื่อปลด File Lock
    try:
        import gc
        gc.collect()
        print("[vector_store] 🗑️ Garbage collection done (DB Lock released).")
    except Exception as e:
        print(f"[vector_store] ❌ GC Error: {e}")

def _normalize_metadata(md: dict) -> dict:
    """
    ปรับรูปแบบข้อมูลเมตา (Metadata Normalization) 
    แปลงค่าเชิงซ้อนให้เป็นสตริง (String) เพื่อให้สามารถบันทึกลง ChromaDB ได้อย่างปลอดภัย
    """
    simple: dict = {}
    for k, v in (md or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            simple[k] = v
        else:
            try:
                simple[k] = str(v)
            except Exception:
                simple[k] = repr(v)
    return simple

# -----------------------------------------------------------
# Process 1: Document Indexing (จัดเก็บและสร้างดัชนีข้อมูล)
# -----------------------------------------------------------
def index_chunks(
    chunks: List[Chunk],
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    จัดเก็บข้อมูลกลุ่มย่อย (Chunks) เข้าสู่ Vector Database พร้อม Metadata 
    สำหรับการค้นหาขั้นสูง
    """
    if not chunks:
        return

    vectordb = get_vector_store(persist_directory, collection_name)
    texts = [c.content for c in chunks]
    
    raw_metadatas = [
        (c.metadata or {}) | {
            "doc_id": sanitize_doc_id(c.doc_id),  
            "doc_type": c.doc_type,
            "source": c.source,
            "page": c.page,
            "chunk_id": c.id,
        } for c in chunks
    ]
    
    metadatas = [_normalize_metadata(md) for md in raw_metadatas]
    ids = [c.id for c in chunks]

    try:
        logger.info(f"[vector_store] Indexing {len(chunks)} chunks...")
        vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        unique_doc_ids = set(md["doc_id"] for md in raw_metadatas)
        logger.info(f"[vector_store] Indexed doc_ids: {unique_doc_ids}")
        
        try:
            vectordb.persist()
        except Exception: 
            pass
    except Exception as e:
        logger.exception("[vector_store] Indexing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Indexing error: {e}") from e


# -----------------------------------------------------------
# Process 2: Retrieval (ค้นหาและสกัดข้อมูล)
# -----------------------------------------------------------
def _python_filter_documents(
    raw_docs: List[Document], 
    doc_ids: Optional[List[str]], 
    sources: Optional[List[str]], 
    doc_types: Optional[List[str]]
) -> List[Document]:
    """
    กระบวนการกรองข้อมูลส่วนเสริม (Application-layer Filtering)
    ทำงานร่วมกับกระบวนการค้นหาหลักเพื่อรองรับเงื่อนไขการค้นหาที่ซับซ้อน
    """
    filtered = []
    
    sanitized_doc_ids = None
    if doc_ids:
        sanitized_doc_ids = set(sanitize_doc_id(d) for d in doc_ids)
    
    for d in raw_docs:
        md = d.metadata or {}
        
        if not filtered:
            logger.debug(f"[vector_store] Sample metadata: {md}")
        
        # กรองข้อมูลผ่านรหัสเอกสาร (Document ID Validation)
        if sanitized_doc_ids:
            found_id = md.get("doc_id")
            if not found_id:
                continue
            
            normalized_found_id = sanitize_doc_id(str(found_id))
            if normalized_found_id not in sanitized_doc_ids:
                continue
                
        # กรองข้อมูลผ่านประเภทแหล่งที่มา (Source Validation)
        if sources:
            doc_source = md.get("source")
            if not doc_source or str(doc_source) not in sources:
                continue
            
        # กรองข้อมูลผ่านประเภทเอกสาร (Document Type Validation)
        if doc_types:
            doc_type = md.get("doc_type")
            if not doc_type or str(doc_type) not in doc_types:
                continue
            
        filtered.append(d)
    
    return filtered


def search_similar(
    query: str,
    k: int = 5,
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    doc_ids: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    doc_types: Optional[List[str]] = None,
) -> List[Document]:
    """
    ระบบค้นหาข้อมูลอัจฉริยะ (Smart Similarity Search)
    รองรับกลไกการกรองระดับฐานข้อมูล (Native Filter) และระบบคัดกรองส่วนเสริมแบบอัตโนมัติ
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    vectordb = get_vector_store(persist_directory, collection_name)

    sanitized_doc_ids = None
    if doc_ids:
        sanitized_doc_ids = [sanitize_doc_id(d) for d in doc_ids if d]
        logger.info(f"[vector_store] Original doc_ids: {doc_ids} -> Sanitized: {sanitized_doc_ids}")

    # --- 1. เตรียมเงื่อนไขสำหรับการกรองระดับฐานข้อมูล (Native Filtering Strategy) ---
    where_filter = {}
    
    if sanitized_doc_ids:
        if len(sanitized_doc_ids) == 1:
            where_filter["doc_id"] = sanitized_doc_ids[0]
        else:
            where_filter = None  
            
    if sources:
        if len(sources) == 1:
            if where_filter is not None:
                where_filter["source"] = sources[0]
        else:
            where_filter = None  
            
    if doc_types:
        if len(doc_types) == 1:
            if where_filter is not None:
                where_filter["doc_type"] = doc_types[0]
        else:
            where_filter = None

    # --- 2. ดำเนินการค้นหา (Execution & Fallback Strategy) ---
    try:
        use_native_filter = where_filter is not None and where_filter != {}
        
        if use_native_filter:
            logger.info(f"[vector_store] Using NATIVE filter: {where_filter}")
            results = vectordb.similarity_search(query, k=k, filter=where_filter)
            
            if not results:
                logger.warning(f"[vector_store] Native filter returned 0 results. Switching to Python filter.")
                use_native_filter = False  
        
        # กรณีฐานข้อมูลไม่รองรับเงื่อนไขซับซ้อน จะสลับมาใช้กระบวนการกรองส่วนเสริมแทน
        if not use_native_filter:
            logger.info(f"[vector_store] Using PYTHON filter for: doc_ids={sanitized_doc_ids}, sources={sources}, doc_types={doc_types}")
            
            fetch_size = max(k * 20, 200)  # [FIX] เพิ่ม fetch size เพื่อดึงข้อมูลให้มากพอก่อน filter
            raw_docs = vectordb.similarity_search(query, k=fetch_size)
            
            logger.info(f"[vector_store] Fetched {len(raw_docs)} raw documents for Python filtering")
            
            if raw_docs:
                found_doc_ids = set(d.metadata.get("doc_id") for d in raw_docs if d.metadata)
                logger.info(f"[vector_store] Available doc_ids in fetched results: {found_doc_ids}")
            
            results = _python_filter_documents(raw_docs, doc_ids, sources, doc_types)[:k]
        
        logger.info(f"[vector_store] Search query='{query[:50]}...' returned {len(results)} results")
        
        return results

    except Exception as e:
        # กลไกการฟื้นฟูระบบอัตโนมัติ (Auto-healing) เมื่อเกิดข้อผิดพลาดรุนแรงระดับฐานข้อมูล
        error_msg = str(e)
        logger.warning(f"[vector_store] Search Exception: {error_msg}")

        is_db_corruption = (
            "Nothing found on disk" in error_msg 
            or "InternalError" in error_msg 
            or "segment reader" in error_msg
            or "sqlite" in error_msg.lower()
            or "Error finding id" in error_msg 
        )

        if is_db_corruption or isinstance(e, ChromaInternalError):
            logger.warning("[vector_store] 🚨 DB Corruption/Change detected. Reloading Vector Store...")
            
            # บังคับรีโหลดและสร้างการเชื่อมต่อฐานข้อมูลใหม่
            vectordb = get_vector_store(persist_directory, collection_name, reload=True)
            
            try:
                raw_docs = vectordb.similarity_search(query, k=k*10)
                results = _python_filter_documents(raw_docs, doc_ids, sources, doc_types)[:k]
                logger.info(f"[vector_store] Retry success. Found {len(results)} results.")
                return results
            except Exception as final_e:
                logger.error(f"[vector_store] Retry failed: {final_e}")
                return []
        
        raise e


# -----------------------------------------------------------
# Debugging & Inspection Utilities
# -----------------------------------------------------------
def get_collection_info(
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Dict:
    """
    ระบบวิเคราะห์และตรวจสอบโครงสร้างภายใน (Collection Inspection) 
    สำหรับดูข้อมูลเชิงสถิติของ Vector Database แบบเรียลไทม์
    """
    try:
        vectordb = get_vector_store(persist_directory, collection_name, reload=True)
        
        sample_docs = vectordb.similarity_search("test", k=5)
        
        doc_ids = set()
        sources = set()
        doc_types = set()
        
        for doc in sample_docs:
            md = doc.metadata or {}
            if md.get("doc_id"):
                doc_ids.add(md.get("doc_id"))
            if md.get("source"):
                sources.add(md.get("source"))
            if md.get("doc_type"):
                doc_types.add(md.get("doc_type"))
        
        return {
            "collection_name": collection_name,
            "sample_count": len(sample_docs),
            "unique_doc_ids": list(doc_ids),
            "unique_sources": list(sources),
            "unique_doc_types": list(doc_types),
            "sample_metadata": [doc.metadata for doc in sample_docs[:3]]
        }
    except Exception as e:
        logger.exception("[vector_store] Failed to get collection info: %s", e)
        return {"error": str(e)}