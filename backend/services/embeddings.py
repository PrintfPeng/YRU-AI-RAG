# backend/services/embeddings.py
from __future__ import annotations

import os
import time
import logging
import requests
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Embedding Model Configuration
# -------------------------------------------------------------------
_EMBEDDING_MODEL_NAME = "bge-m3:latest"

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


class OllamaDirectEmbeddings(Embeddings):
    """
    ใช้ Ollama native /api/embeddings แทน /v1/embeddings
    เพราะ /v1/embeddings ทำให้ bge-m3 llama-server crash เป็นประจำ
    """

    def __init__(self, base_url: str, model: str, max_retries: int = 2, timeout: int = 150):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        # ตัด /v1 ออกถ้ามี แล้วสร้าง URL ไปยัง native endpoint
        ollama_base = base_url.rstrip('/')
        if ollama_base.endswith('/v1'):
            ollama_base = ollama_base[:-3]
        self._embed_url = f"{ollama_base}/api/embeddings"
        logger.info(f"[OllamaEmbed] Using native endpoint: {self._embed_url} model={model}")

    def _embed_single(self, text: str) -> List[float]:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self._embed_url,
                    # keep_alive=-1 keeps bge-m3 pinned in GPU memory indefinitely
                    json={"model": self.model, "prompt": text, "keep_alive": -1},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding", [])
                if not emb:
                    raise ValueError(f"Empty embedding: {data}")
                return emb
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"[OllamaEmbed] attempt {attempt+1} failed: {e}. retry in {wait}s")
                    time.sleep(wait)
        raise RuntimeError(f"Embedding failed after {self.max_retries} attempts: {last_err}")

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i, text in enumerate(texts):
            results.append(self._embed_single(text))
            if (i + 1) % 10 == 0:
                logger.info(f"[OllamaEmbed] {i+1}/{len(texts)} embedded")
        return results


def get_embedding_model():
    """
    สร้างและคืนค่า Embedding Client แบบ Singleton
    """
    global _embeddings_client

    if _embeddings_client is None:
        load_dotenv()
        base_url = os.getenv("OPEN_WEBUI_BASE_URL", "http://172.19.0.1:11434/v1")
        model = os.getenv("EMBEDDING_MODEL", _EMBEDDING_MODEL_NAME)

        print(f"⏳ Initializing OllamaDirectEmbeddings: {model} via {base_url}")

        _embeddings_client = OllamaDirectEmbeddings(
            base_url=base_url,
            model=model,
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