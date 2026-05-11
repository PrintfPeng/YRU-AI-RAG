# backend/services/embeddings.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------------------------------------------
# Embedding Model Configuration
# -------------------------------------------------------------------
# เลือกใช้โมเดล Embeddings แบบ Local (HuggingFace) แทนการใช้ API ภายนอก
# เพื่อลดค่าใช้จ่าย (Cost-effective) และเพิ่มความปลอดภัยของข้อมูล (Data Privacy)
# โมเดล 'intfloat/multilingual-e5-large' ถูกเลือกเพราะรองรับภาษาไทยได้ดีเยี่ยมและให้ความแม่นยำสูง
_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" 
# ทางเลือกสำรองสำหรับเครื่องที่มีทรัพยากรจำกัด: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# -------------------------------------------------------------------
# Singleton Instance สำหรับจัดการหน่วยความจำ
# -------------------------------------------------------------------
# เก็บ Instance ของโมเดลไว้ในหน่วยความจำเพื่อหลีกเลี่ยงการโหลดโมเดลใหม่ซ้ำๆ ทุกครั้งที่มีการเรียกใช้
_embeddings_client: HuggingFaceEmbeddings | None = None


# =============================================================================
# DATA INTEGRITY CONTRACT (ข้อตกลงในการรักษาความสมบูรณ์ของข้อมูล)
# =============================================================================
# 1. เลเยอร์นี้มีหน้าที่สร้าง Vector Embeddings จากข้อมูลส่วนที่เป็น 'text' เท่านั้น
# 2. ฟังก์ชันในเลเยอร์นี้ "ห้าม" ดัดแปลง, ปรับมาตรฐาน (Normalize) หรือเปลี่ยนแปลง 'metadata' โดยเด็ดขาด
# 3. ข้อมูล Metadata ทุกชนิด (เช่น 'source': 'table', ฟิลด์ JSON, หรือโครงสร้าง HTML) 
#    จะต้องถูกส่งผ่าน (Passthrough) ไปยัง Vector Store อย่างครบถ้วน 100%
# 4. ลอจิกใดๆ ที่เกี่ยวข้องกับการจัดการโครงสร้างตารางหรือรูปภาพ ต้องทำที่ chunking.py เท่านั้น
# =============================================================================


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    สร้างและคืนค่า Embedding Client แบบ Singleton
    
    Returns:
        HuggingFaceEmbeddings: อินสแตนซ์ของโมเดลที่พร้อมใช้งาน
    """
    global _embeddings_client

    if _embeddings_client is None:
        print(f"⏳ Loading Local Embedding Model: {_EMBEDDING_MODEL_NAME} ...")
        
        # กำหนดค่าตัวแปรของโมเดล (ใช้อุปกรณ์ CPU เป็นค่าเริ่มต้น แต่สามารถปรับเป็น 'cuda' ได้หากมี GPU)
        # encode_kwargs={'normalize_embeddings': True} ช่วยให้การคำนวณ Cosine Similarity แม่นยำขึ้น
        _embeddings_client = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )

    return _embeddings_client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    สร้าง Vector Embeddings แบบกลุ่ม (Batch Processing) สำหรับข้อความหลายชุด
    
    Args:
        texts (List[str]): รายการข้อความที่ต้องการแปลง
        
    Returns:
        List[List[float]]: รายการของเวกเตอร์ที่สอดคล้องกับข้อความ
        
    หมายเหตุ: ฟังก์ชันที่เรียกใช้งานต้องจัดการผูก Metadata เข้ากับผลลัพธ์ด้วยตนเอง
    """
    if not texts:
        return []

    client = get_embedding_model()
    return client.embed_documents(texts)


def embed_query(text: str) -> List[float]:
    """
    สร้าง Vector Embedding สำหรับข้อความเดี่ยว 
    (มักใช้ในการแปลงคำถามของผู้ใช้เพื่อนำไปทำ Similarity Search)
    
    Args:
        text (str): คำถามหรือข้อความเดี่ยว
        
    Returns:
        List[float]: เวกเตอร์ของข้อความนั้น
    """
    client = get_embedding_model()
    return client.embed_query(text)


def embed_with_metadata(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    สร้าง Vector Embeddings โดยรับประกันว่า Metadata จะไม่สูญหายหรือถูกดัดแปลง (Safe Helper)
    
    Args:
        items: List ของ Dictionary ที่ประกอบด้วย {"text": str, "metadata": dict}
               (อนุญาตให้มีฟิลด์เพิ่มเติมใน metadata และจะถูกเก็บรักษาไว้อย่างดี)

    Returns:
        List ของ Dictionary ในรูปแบบ: [{"embedding": [...], "metadata": {...}}, ...]

    ขั้นตอนการทำงาน:
    1. สกัดเฉพาะ 'text' ออกมาเพื่อนำไปสร้างเวกเตอร์
    2. ทำการแปลงเป็นเวกเตอร์แบบกลุ่ม (Batch) เพื่อประสิทธิภาพสูงสุด
    3. นำ Metadata ต้นฉบับมาประกอบร่างคืนโดยไม่มีการแตะต้องเนื้อหาภายใน (Blind Passthrough)
    """
    if not items:
        return []

    # 1. สกัดข้อความอย่างระมัดระวัง (หากไม่มีค่า text จะใช้ string ว่างเพื่อป้องกันระบบล่ม)
    texts = [str(item.get("text", "")) for item in items]

    # 2. สร้างเวกเตอร์สำหรับข้อความทั้งหมดในครั้งเดียว
    vectors = embed_texts(texts)

    # 3. ประกอบร่างเวกเตอร์เข้ากับ Metadata เดิม
    results = []
    for i, vector in enumerate(vectors):
        original_meta = items[i].get("metadata", {})
        
        # การันตีความปลอดภัย: Metadata แหล่งที่มา (เช่น โครงสร้างตาราง) จะถูกส่งผ่านแบบ 1-to-1
        results.append({
            "embedding": vector,
            "metadata": original_meta  # การส่งค่าผ่าน Reference ถือว่าปลอดภัยและประหยัดหน่วยความจำ
        })

    return results