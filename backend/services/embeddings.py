# backend/services/embeddings.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# -------------------------------------------------------------------
# Embedding Model Configuration
# -------------------------------------------------------------------
# ใช้โมเดล Embedding ที่โฮสต์อยู่บน Open WebUI Server ของคุณ
_EMBEDDING_MODEL_NAME = "bge-m3"

# -------------------------------------------------------------------
# Singleton Instance สำหรับจัดการหน่วยความจำ
# -------------------------------------------------------------------
_embeddings_client = None

# =============================================================================
# DATA INTEGRITY CONTRACT (ข้อตกลงในการรักษาความสมบูรณ์ของข้อมูล)
# =============================================================================
# 1. เลเยอร์นี้มีหน้าที่สร้าง Vector Embeddings จากข้อมูลส่วนที่เป็น 'text' เท่านั้น
# 2. ฟังก์ชันในเลเยอร์นี้ "ห้าม" ดัดแปลง, ปรับมาตรฐาน (Normalize) หรือเปลี่ยนแปลง 'metadata' โดยเด็ดขาด
# 3. ข้อมูล Metadata ทุกชนิด จะต้องถูกส่งผ่าน (Passthrough) ไปยัง Vector Store อย่างครบถ้วน
# =============================================================================

def get_embedding_model():
    """
    สร้างและคืนค่า Embedding Client แบบ Singleton
    """
    global _embeddings_client

    if _embeddings_client is None:
        load_dotenv()
        # ดึงค่า URL และ API Key ของ Open WebUI จากไฟล์ .env
        base_url = os.getenv("OPEN_WEBUI_BASE_URL", "http://10.10.2.154:3000/api/v1")
        api_key = os.getenv("OPEN_WEBUI_API_KEY", "sk-f6f4029b19cd4092bddbbfa6bb708102")
        
        print(f"⏳ Connecting to Remote Embedding Model: {_EMBEDDING_MODEL_NAME} at {base_url} ...")
        
        _embeddings_client = OpenAIEmbeddings(
            model=_EMBEDDING_MODEL_NAME,
            base_url=base_url,
            api_key=api_key,
            check_embedding_ctx_length=False
        )

    return _embeddings_client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    สร้าง Vector Embeddings แบบกลุ่ม (Batch Processing) สำหรับข้อความหลายชุด
    """
    if not texts:
        return []

    client = get_embedding_model()
    return client.embed_documents(texts)


def embed_query(text: str) -> List[float]:
    """
    สร้าง Vector Embedding สำหรับข้อความเดี่ยว 
    (มักใช้ในการแปลงคำถามของผู้ใช้เพื่อนำไปทำ Similarity Search)
    """
    client = get_embedding_model()
    return client.embed_query(text)


def embed_with_metadata(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    สร้าง Vector Embeddings โดยรับประกันว่า Metadata จะไม่สูญหายหรือถูกดัดแปลง (Safe Helper)
    """
    if not items:
        return []

    # 1. สกัดเฉพาะ 'text' ออกมา
    texts = [str(item.get("text", "")) for item in items]

    # 2. แปลงเป็นเวกเตอร์ผ่าน Open WebUI
    vectors = embed_texts(texts)

    # 3. ประกอบร่างคืน
    results = []
    for i, vector in enumerate(vectors):
        original_meta = items[i].get("metadata", {})
        results.append({
            "embedding": vector,
            "metadata": original_meta
        })

    return results