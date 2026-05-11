# backend/services/rag.py
from __future__ import annotations

import os
import re
import json
import logging
import math
import collections
import tiktoken
from typing import Dict, List, Optional
from pathlib import Path
from difflib import SequenceMatcher

from dotenv import load_dotenv

# -------------------------------------------------------------------
# LLM Providers & Core Dependencies
# -------------------------------------------------------------------
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    _HAS_GENAI = True
except Exception:
    ChatOpenAI = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    _HAS_GENAI = False

try:
    from sentence_transformers import CrossEncoder
    _HAS_RERANKER = True
    _RERANK_MODEL = None  # Lazy load implementation for performance
except ImportError:
    _HAS_RERANKER = False
    _RERANK_MODEL = None

from .vector_store import search_similar
from .llm_provider import LocalLLMProvider

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# System Paths & Environments
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGESTED_DIR = PROJECT_ROOT / "ingested"

# Q&A Matching Cache
_QNA_CACHE: Dict[str, List[Dict[str, str]]] = {}
_QNA_CACHE_MAX_SIZE = 100

load_dotenv(override=True)

# -------------------------------------------------------------------
# Application Configuration
# -------------------------------------------------------------------
_CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
_CUSTOM_API_BASE = os.getenv("CUSTOM_API_BASE")

# กำหนดชื่อโมเดลหลักที่ใช้ในการประมวลผล (Primary Model)
_LL_MODEL_FAST = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")
_LL_MODEL_SMALL = _LL_MODEL_FAST 

# ลดค่า Temperature เพื่อให้คำตอบมีความแม่นยำและเป็นเหตุเป็นผลสูงสุด (ลด Hallucination)
_DEFAULT_TEMPERATURE = 0.1 

# โมเดลสำหรับจัดอันดับข้อมูลซ้ำ (Cross-Encoder Re-ranking Model)
_RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# การตั้งค่าเกณฑ์คัดกรองความแม่นยำ (Threshold Configurations)
MIN_SCORE_THRESHOLD = 0.25 # เกณฑ์ความมั่นใจขั้นต่ำของเอกสาร
MIN_KEYWORD_OVERLAP = 1    # ต้องมีคำสำคัญตรงกันอย่างน้อย 1 คำ สำหรับคำถามที่มีความยาว

INTENT_THRESHOLDS = {
    "qna_match": 0.20,
    "table": 0.15,
    "text": 0.10,
    "both": 0.15
}

# Regular Expression สำหรับตรวจจับรูปแบบ ถาม-ตอบ (Q&A Extraction)
_QNA_PATTERN = re.compile(
    r"(?:\d+\s*[\.\-\)]\s*)?"
    r"(?:ถาม|q|question)\s*[:\-]?\s*"
    r"(?P<q>.+?)\s*"
    r"(?:ตอบ|a|answer)\s*[:\-]?\s*"
    r"(?P<a>.+?)(?=(?:\d+\s*[\.\-\)]\s*)?(?:ถาม|q|question)\s*[:\-]?|\Z)",
    re.IGNORECASE | re.DOTALL,
)

def normalize_score(raw_score: float) -> float:
    """แปลงคะแนนดิบให้อยู่ในรูปแบบความน่าจะเป็น (Sigmoid Normalization)"""
    try:
        return 1 / (1 + math.exp(-raw_score))
    except OverflowError:
        return 0.0 if raw_score < 0 else 1.0


def _count_tokens(text: str, model: str = "gpt-4") -> int:
    """นับจำนวน Token จริงของข้อความตามมาตรฐานของโมเดล"""
    try:
        # ใช้ encoding ของ gpt-4 เป็นมาตรฐาน (ซึ่งครอบคลุมถึง Qwen และโมเดลยุคใหม่ส่วนใหญ่)
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback กรณี Error ให้กะเกณฑ์แบบหยาบๆ (ตัวอักษร / 3)
        return len(text) // 3

# -------------------------------------------------------------------
# Helper: Data Sanitization
# -------------------------------------------------------------------
def sanitize_doc_id(doc_id: str) -> str:
    """
    จัดการและทำความสะอาด Document ID ให้สอดคล้องกับฐานข้อมูล 
    และรองรับชุดอักขระภาษาไทยเพื่อความเสถียรของระบบ
    """
    if not doc_id:
        return ""
    doc_id = doc_id.lower().strip()
    doc_id = re.sub(r'\s+', '_', doc_id)
    doc_id = re.sub(r'[^a-z0-9_\u0E00-\u0E7F-]', '', doc_id) 
    return doc_id


def _sanitize_html_content(html: str) -> str:
    """ลบส่วนประกอบที่อาจเป็นอันตรายออกจาก HTML เพื่อป้องกันการแทรกแซง (XSS Prevention)"""
    if not html: return ""
    html = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r" on\w+=", " data-blocked-event=", html, flags=re.IGNORECASE)
    html = re.sub(r"javascript:", "blocked:", html, flags=re.IGNORECASE)
    return html


# -------------------------------------------------------------------
# Helper: LLM Provider Initialization
# -------------------------------------------------------------------
def _get_llm_instance(model: Optional[str] = None, temperature: float = _DEFAULT_TEMPERATURE):
    """สร้างอินสแตนซ์ของ LLM ตัวหลักผ่านระบบ LocalLLMProvider"""
    if not _HAS_GENAI:
        logger.debug("[rag] langchain_openai not installed -> no LLM available")
        return None
    
    try:
        return LocalLLMProvider.get_primary_llm(temperature=temperature)
    except Exception as e:
        logger.exception("[rag] Failed to init LLM: %s", e)
        return None


def _get_google_llm():
    """สร้างอินสแตนซ์ของ Google Gemini เพื่อใช้เป็นระบบประมวลผลสำรอง (Fallback LLM)"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not ChatGoogleGenerativeAI:
        return None
    
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=2048,
            convert_system_message_to_human=True 
        )
    except Exception as e:
        logger.error(f"[rag] Failed to init Google LLM: {e}")
        return None


# -------------------------------------------------------------------
# Helper: Re-ranker Initialization
# -------------------------------------------------------------------
def _get_reranker_model():
    """โหลดโมเดล Cross-Encoder เมื่อมีการเรียกใช้งานครั้งแรก (Lazy Initialization)"""
    global _RERANK_MODEL
    if not _HAS_RERANKER:
        return None
    
    if _RERANK_MODEL is None:
        try:
            logger.info(f"[rag] Loading Re-ranking model: {_RERANK_MODEL_NAME}")
            _RERANK_MODEL = CrossEncoder(_RERANK_MODEL_NAME, max_length=512)
        except Exception as e:
            logger.error(f"[rag] Failed to load reranker: {e}")
            return None
    return _RERANK_MODEL


# -------------------------------------------------------------------
# Guardrails & Intent Classification
# -------------------------------------------------------------------
def _rule_based_intent(query: str) -> Optional[str]:
    """วิเคราะห์และจัดหมวดหมู่คำถามผ่านระบบคำสำคัญ (Rule-based Keyword Extraction)"""
    if not query or not query.strip(): return None
    q = query.lower()
    table_keywords = ["ตาราง", "table", "คอลัมน์", "column", "แถว", "row", "สรุป", "summary", "ยอด", "amount", "list", "รายการ", "schedule"]
    image_keywords = ["รูป", "รูปภาพ", "image", "logo", "กราฟ", "graph", "chart", "diagram", "photo", "ภาพ"]
    is_table = any(w in q for w in table_keywords)
    is_image = any(w in q for w in image_keywords)
    if is_table and not is_image: return "table"
    if is_image and not is_table: return "both"
    if is_table and is_image: return "both"
    return "text"

def _detect_general_intent(query: str) -> bool:
    """กรองคำถามทั่วไปหรือการทักทาย เพื่อหลีกเลี่ยงการค้นหาเอกสารโดยไม่จำเป็น (Optimization)"""
    q = query.lower().strip()
    general_keywords = ["สวัสดี", "hello", "hi", "วันนี้วันอะไร", "อากาศ", "who are you", "คุณคือใคร", "สบายดีไหม"]
    if q in general_keywords:
        return True
    if "วันนี้" in q and "วันอะไร" in q:
        return True
    return False

def _keyword_overlap_count(query: str, text: str) -> int:
    """ประเมินความสอดคล้องเบื้องต้นโดยใช้วิธีนับจุดตัดของคำสำคัญ (Keyword Intersection)"""
    q_clean = re.sub(r'[^\w\s]', '', query).lower()
    t_clean = re.sub(r'[^\w\s]', '', text).lower()
    
    q_tokens = set(q_clean.split())
    t_tokens = set(t_clean.split())
    
    stopwords = {"คือ", "เป็น", "อยู่", "จะ", "ได้", "ที่", "ซึ่ง", "อัน", "ของ", "what", "is", "are", "the", "a", "an", "ครับ", "ค่ะ"}
    q_tokens = q_tokens - stopwords
    
    if not q_tokens: return 0
    return len(q_tokens.intersection(t_tokens))

def _filter_relevant_docs(query: str, docs: list, min_score: float = MIN_SCORE_THRESHOLD) -> list:
    """
    คัดกรองเอกสารอย่างเข้มงวด ป้องกันการส่งข้อมูลที่ไม่เกี่ยวข้องให้ LLM 
    เพื่อลดปัญหาอาการหลอนของโมเดล (Strict Guardrails for Anti-Hallucination)
    """
    passed = []
    for d in docs:
        score = d.metadata.get("ai_score", 0.0)
        content = d.page_content or ""
        
        # Guard 1: ตรวจสอบความมั่นใจขั้นต่ำ (Score Threshold)
        if score < min_score:
            continue
            
        # Guard 2: ตรวจสอบความเชื่อมโยงของคำสำคัญกรณีคำถามมีความยาว (Keyword Validation)
        if len(query) > 10: 
            overlap = _keyword_overlap_count(query, content)
            if overlap < MIN_KEYWORD_OVERLAP:
                # อนุโลมให้ผ่านหากคะแนนความหมาย (Semantic Score) สูง แม้จะไม่พบคำค้นหาตรงตัว
                if score < 0.75: 
                    continue

        passed.append(d)
    return passed


# -------------------------------------------------------------------
# Context Building & Formatting
# -------------------------------------------------------------------
def _build_context_text(docs) -> str:
    """เตรียมข้อมูลสำหรับการป้อนเข้าสู่ Context Window โดยคำนวณ Token จริง"""
    parts: List[str] = []
    current_tokens = 0
    
    # ตั้งค่าขีดจำกัดสูงสุด (เช่น 3500 tokens เพื่อเผื่อที่ไว้ให้ Prompt และคำตอบ)
    MAX_CONTEXT_TOKENS = 3500 

    parts.append("⚠️ **แหล่งข้อมูลอ้างอิง:** (เรียงตามความเกี่ยวข้อง)\n")

    for i, d in enumerate(docs, 1):
        content = getattr(d, "page_content", "") or getattr(d, "content", "") or ""
        content = content.replace("\x00", "") 
        
        md = d.metadata or {}
        doc_id = md.get("doc_id", "unknown")
        page = md.get("page", "?")
        score = md.get("ai_score", 0.0)
        
        # สร้างเนื้อหาของ Source นี้
        header = f"[SOURCE {i}] ID: {doc_id} | Page: {page} | Score: {score:.2f}"
        chunk_text = f"{header}\n{content}\n\n"
        
        # นับ Token ของ Chunk นี้
        chunk_tokens = _count_tokens(chunk_text)
        
        # ถ้าบวกเข้าไปแล้วเกินขีดจำกัด ให้หยุดเติม Context
        if current_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(f"[rag] Context limit reached. Skipping remaining {len(docs) - i + 1} docs.")
            break

        parts.append(chunk_text)
        current_tokens += chunk_tokens

    return "".join(parts)


def _generate_fallback_answer(docs, error_msg: str = "") -> str:
    """สร้างคำตอบจากข้อมูลดิบโดยตรง กรณีที่ระบบ LLM ขัดข้องทั้งหมด (Zero-Downtime Guarantee)"""
    if not docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร (และ AI ไม่สามารถประมวลผลได้ในขณะนี้)"

    snippets = []
    for i, d in enumerate(docs[:4], 1):
        content = getattr(d, "page_content", "") or getattr(d, "content", "") or ""
        md = d.metadata or {}
        page = md.get('page', '?')
        snippet_text = content[:400].replace("\n", " ").strip() + "..."
        snippets.append(f"**{i}. (หน้า {page})** {snippet_text}")
    
    joined_snippets = "\n\n".join(snippets)
    header = f"⚠️ **แจ้งเตือน ({error_msg}):** ระบบจึงดึงเนื้อหาที่เกี่ยวข้องจากเอกสารมาแสดงให้โดยตรงครับ:\n\n"
              
    return header + joined_snippets


def _filter_table_docs_by_category(docs, query: str):
    return docs


# -------------------------------------------------------------------
# Advanced Re-ranking Logic (Semantic Matching Engine)
# -------------------------------------------------------------------
def _clean_text_for_rerank(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:1000]

def _rerank_documents(query: str, docs: list, top_k: int) -> list:
    """
    จัดอันดับความเกี่ยวข้องของเอกสารใหม่ ผ่านการให้คะแนนแบบผสมผสาน 
    (Hybrid Scoring: Keyword Boosting + Cross-Encoder Semantic Matching + Intent Penalty)
    """
    if not docs:
        return []

    # 1. การเพิ่มน้ำหนักโดยอิงจากคำสำคัญ (Keyword Boosting Phase)
    query_terms = query.lower().split()
    scored_docs = []
    for d in docs:
        content = (getattr(d, "page_content", "") or "").lower()
        base_score = 0.0
        
        for term in query_terms:
            if term in content:
                base_score += 1.0
        
        if query.lower() in content:
            base_score += 3.0
            
        # ปรับลดคะแนนเอกสารที่ไม่ตรงกับเจตนาของคำถาม (Intent Penalty Rules)
        source_type = str(getattr(d, "metadata", {}).get("source", "text")).lower()
        query_lower = query.lower()
        
        is_img_q = any(x in query_lower for x in ["รูปภาพ", "image", "logo", "กราฟ", "แผนภูมิ", "ถ่ายรูป"])
        is_tbl_q = any(x in query_lower for x in ["ตาราง", "table", "แบบฟอร์ม", "สถิติ"])
        
        if is_img_q:
            if source_type != "image":
                base_score *= 0.1  # ปรับลดคะแนนลงอย่างหนัก หากถามรูปแต่ได้ข้อความ
        elif is_tbl_q:
            if source_type != "table":
                base_score *= 0.2  # ปรับลดคะแนน หากถามตารางแต่ได้ข้อมูลประเภทอื่น
        else:
            if source_type == "image":
                base_score *= 0.1  # ลดความสำคัญรูปภาพ หากคำถามเป็นบริบทข้อความทั่วไป
            elif source_type == "table":
                base_score *= 0.5  
                
        if "ai_score" not in d.metadata:
            d.metadata["ai_score"] = 0.0
            
        d.metadata["keyword_score"] = base_score
        scored_docs.append(d)

    # 2. การจัดอันดับเชิงความหมาย (Cross-Encoder AI Re-ranking Phase)
    reranker = _get_reranker_model()
    if reranker:
        try:
            valid_pairs_indices = []
            pairs = []
            
            for i, doc in enumerate(scored_docs):
                clean_content = _clean_text_for_rerank(doc.page_content)
                if clean_content:
                    pairs.append([query, clean_content])
                    valid_pairs_indices.append(i)
            
            if pairs:
                raw_scores = reranker.predict(pairs)
                
                for idx, raw in zip(valid_pairs_indices, raw_scores):
                    norm_score = normalize_score(float(raw))
                    
                    # บังคับใช้กฎ Intent Penalty กับคะแนน AI เช่นเดียวกัน
                    source_type = str(scored_docs[idx].metadata.get("source", "text")).lower()
                    
                    if is_img_q:
                        if source_type != "image":
                            norm_score *= 0.1
                    elif is_tbl_q:
                        if source_type != "table":
                            norm_score *= 0.2
                    else:
                        if source_type == "image":
                            norm_score *= 0.1
                        elif source_type == "table":
                            norm_score *= 0.5
                        
                    scored_docs[idx].metadata["ai_score"] = norm_score
                    scored_docs[idx].metadata["raw_score"] = float(raw)
                
                # จัดเรียงลำดับใหม่ตามคะแนนประเมินเชิงความหมาย
                scored_docs.sort(key=lambda x: x.metadata["ai_score"], reverse=True)
                return scored_docs[:top_k]

        except Exception as e:
            logger.warning(f"[rag] Re-ranking failed: {e}")

    # 3. กระบวนการย้อนกลับกรณีโมเดล AI ขัดข้อง (Fallback to Keyword Sort)
    scored_docs.sort(key=lambda x: x.metadata.get("keyword_score", 0), reverse=True)
    for d in scored_docs:
        if d.metadata["ai_score"] == 0.0:
            d.metadata["ai_score"] = 0.3 
    
    return scored_docs[:top_k]


# -------------------------------------------------------------------
# Q&A Extraction and Matching
# -------------------------------------------------------------------
def _load_qna_pairs_for_doc(doc_id: str) -> List[Dict[str, str]]:
    """สกัดคู่คำถาม-คำตอบที่มีอยู่ในเอกสารโดยตรง เพื่อเพิ่มความรวดเร็วในการให้บริการ"""
    if len(_QNA_CACHE) > _QNA_CACHE_MAX_SIZE:
        _QNA_CACHE.clear()

    if doc_id in _QNA_CACHE:
        return _QNA_CACHE[doc_id]

    path = INGESTED_DIR / doc_id / "text.json"
    if not path.exists():
        _QNA_CACHE[doc_id] = []
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _QNA_CACHE[doc_id] = []
        return []

    full = "\n".join((item.get("content") or "") for item in raw)
    pairs: List[Dict[str, str]] = []
    for m in _QNA_PATTERN.finditer(full):
        q = " ".join(m.group("q").split())
        a = " ".join(m.group("a").split())
        if q and a:
            pairs.append({"question": q, "answer": a})
    _QNA_CACHE[doc_id] = pairs
    return pairs

def _simple_similarity(a: str, b: str) -> float:
    """ประเมินความคล้ายคลึงระดับสายอักขระ (String Similarity)"""
    return SequenceMatcher(None, a, b).ratio()

def _find_best_qna_answer_from_docs(query: str, docs) -> Optional[Dict]:
    """ค้นหาคำตอบจากรายการ Q&A ที่แคชไว้ เพื่อตอบกลับในทันทีหากมีความสอดคล้องสูง"""
    qna_doc_ids = sorted({
        (d.metadata or {}).get("doc_id")
        for d in docs
        if (d.metadata or {}).get("doc_id") 
    })
    qna_doc_ids = [d for d in qna_doc_ids if d]
    
    if not qna_doc_ids:
        return None

    all_pairs = []
    for doc_id in qna_doc_ids:
        pairs = _load_qna_pairs_for_doc(doc_id)
        for p in pairs:
            all_pairs.append({"question": p["question"], "answer": p["answer"], "doc_id": doc_id})

    if not all_pairs:
        return None

    best_score = 0.0
    best_item = None
    
    reranker = _get_reranker_model()
    if reranker:
        try:
            input_pairs = [[query, p["question"]] for p in all_pairs]
            raw_scores = reranker.predict(input_pairs)
            
            for i, raw in enumerate(raw_scores):
                norm_score = normalize_score(float(raw))
                if norm_score > best_score:
                    best_score = norm_score
                    best_item = all_pairs[i]
        except Exception:
            pass

    if not best_item:
        for p in all_pairs:
            score = _simple_similarity(query, p["question"])
            if score > best_score:
                best_score = score
                best_item = p
        
    if best_item and best_score >= 0.75: # ดำเนินการตอบเมื่อมีความมั่นใจในระดับสูงเท่านั้น
        return {
            "answer": best_item["answer"],
            "sources": [{"doc_id": best_item["doc_id"], "source": "Q&A Match", "page": "?"}],
            "score": float(best_score)
        }
    return None

# -------------------------------------------------------------------
# Memory & Conversational Flow Control
# -------------------------------------------------------------------
def _get_chat_history(limit: int = 3, current_doc_ids: Optional[List[str]] = None) -> tuple[list[dict], list[str]]:
    """
    [OPTIMIZED] ระบบความจำ: อ่านประวัติการสนทนาย้อนหลังจากท้ายไฟล์ (ประสิทธิภาพสูง)
    ดึงเฉพาะบรรทัดล่าสุดมาตรวจสอบ ไม่โหลดทั้งไฟล์เข้า RAM
    """
    log_path = PROJECT_ROOT / "backend" / "logs" / "qa_log.jsonl"
    history = []
    sticky_doc_ids = []

    if not log_path.exists():
        return history, sticky_doc_ids

    try:
        # ใช้ deque เพื่อเก็บเฉพาะ N บรรทัดสุดท้ายจากไฟล์ (ในที่นี้เอา 50 บรรทัดพอ)
        # วิธีนี้เร็วกว่า readlines() ทั้งไฟล์มหาศาลเมื่อไฟล์มีขนาดใหญ่
        with open(log_path, 'r', encoding='utf-8') as f:
            last_lines = collections.deque(f, maxlen=50) 
        
        lines = list(last_lines)
        if not lines:
            return history, sticky_doc_ids

        # 1. หา Sticky Context (เอกสารล่าสุดที่เคยใช้)
        for line in reversed(lines):
            try:
                data = json.loads(line)
                doc_ids = data.get("doc_ids")
                if doc_ids and isinstance(doc_ids, list):
                    sticky_doc_ids = doc_ids
                    break
            except json.JSONDecodeError:
                continue

        # กำหนดเป้าหมายเอกสาร
        target_doc_ids = current_doc_ids if current_doc_ids else sticky_doc_ids

        # 2. คัดกรองและสกัดบทสนทนาย้อนหลังเฉพาะที่เกี่ยวข้อง
        matched_turns = 0
        for line in reversed(lines):
            try:
                data = json.loads(line)
                log_doc_ids = data.get("doc_ids")
                
                # กรองเอาเฉพาะประวัติที่ใช้เอกสารชุดเดียวกัน
                if target_doc_ids and log_doc_ids != target_doc_ids:
                    continue

                q = data.get("query", "")
                a = data.get("answer", "")
                
                # ไม่เอาประวัติที่เป็น Error หรือหาของไม่เจอมาเป็น Context
                if q and a and "ไม่พบข้อมูล" not in a and "ระบบค้นหาขัดข้อง" not in a:
                    history.insert(0, {"role": "assistant", "content": a})
                    history.insert(0, {"role": "user", "content": q})
                    matched_turns += 1

                if matched_turns >= limit:
                    break
            except json.JSONDecodeError:
                continue

    except Exception as e:
        logger.error(f"[rag] Optimized history read failed: {e}")

    return history, sticky_doc_ids


async def _rewrite_query(query: str, history: list[dict], llm) -> str:
    """ระบบคลายความกำกวม (Query Disambiguation) โดยใช้ LLM วิเคราะห์คำสรรพนามจากบริบทเดิม"""
    if not history or not llm:
        return query

    history_text = "\n".join([f"{item['role']}: {item['content'][:200]}" for item in history])

    sys_prompt = SystemMessage(content=(
        "คุณคือ AI ด่านหน้า มีหน้าที่ 'เรียบเรียงคำถามใหม่' (Query Rewriting)\n"
        "กฎเหล็ก:\n"
        "1. อ่านประวัติการคุย ถ้าคำถามใหม่มีคำสรรพนาม (เช่น เขา, มัน, ที่นั่น, เกรดเท่าไหร่) ให้เปลี่ยนเป็นชื่อคนหรือสิ่งนั้นให้สมบูรณ์\n"
        "2. ห้ามตอบคำถามเด็ดขาด! ให้พิมพ์แค่ 'คำถามที่เรียบเรียงใหม่' ประโยคเดียวเท่านั้น\n"
        "3. ถ้าคำถามใหม่ชัดเจนสมบูรณ์อยู่แล้ว ให้พิมพ์คำถามเดิมเป๊ะๆ กลับมา\n"
    ))
    
    user_prompt = HumanMessage(content=(
        f"=== ประวัติการคุย ===\n{history_text}\n\n"
        f"=== คำถามใหม่ ===\n{query}\n\n"
        f"คำถามที่เรียบเรียงใหม่คือ:"
    ))

    try:
        res = await llm.ainvoke([sys_prompt, user_prompt])
        rewritten = getattr(res, "content", str(res)).strip().strip("'\"")
        return rewritten if rewritten else query
    except Exception as e:
        logger.warning(f"[rag] Query rewrite failed: {e}")
        return query

# -------------------------------------------------------------------
# Core Process: Intelligent RAG Engine
# -------------------------------------------------------------------
async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 10,
    mode: str = "auto",
) -> Dict:
    
    # 1. การตรวจสอบและทำความสะอาด Input ขาเข้า
    if not query or not query.strip():
        return {"answer": "กรุณาพิมพ์คำถามครับ", "sources": [], "intent": None, "mode": mode}

    # 2. เรียกใช้งานระบบจัดการความจำของบทสนทนา
    chat_history, sticky_doc_ids = _get_chat_history(limit=3, current_doc_ids=doc_ids)

    # นำส่งเอกสารต่อเนื่องหากผู้ใช้ไม่ได้เปลี่ยนเอกสาร
    if not doc_ids and sticky_doc_ids:
        doc_ids = sticky_doc_ids
        logger.info(f"[rag] Auto-injected sticky doc_ids: {doc_ids}")

    # 3. เตรียมคำถามและวิเคราะห์เจตนา (Query Preparation & Intent Analysis)
    llm_fast = _get_llm_instance(model=_LL_MODEL_FAST)
    search_query = query 
    if chat_history and llm_fast:
        search_query = await _rewrite_query(query, chat_history, llm_fast)
        logger.info(f"[rag] Rewritten Query: '{query}' -> '{search_query}'")

    if _detect_general_intent(query):
        return {
            "answer": "คำถามนี้ดูเหมือนเป็นคำถามทั่วไป ผมตอบได้เฉพาะข้อมูลที่มีในเอกสารที่แนบมาเท่านั้นครับ (ลองถามเกี่ยวกับเนื้อหาในเอกสารดูนะครับ)",
            "sources": [],
            "intent": "general",
            "mode": mode
        }

    # ระบบคัดกรองโหมดอัตโนมัติ (Deterministic Mode Selector)
    if mode == "auto":
        q_lower = query.lower()
        if any(x in q_lower for x in ["ตาราง", "table", "สถิติ", "แบบฟอร์ม"]):
            intent = "table"
        elif any(x in q_lower for x in ["รูปภาพ", "image", "logo", "โลโก้", "กราฟ", "แผนภูมิ", "แผนภาพ"]):
            intent = "image"
        else:
            intent = "text"
    else:
        intent = mode

    sanitized_doc_ids = None
    if doc_ids:
        sanitized_doc_ids = [sanitize_doc_id(doc_id) for doc_id in doc_ids if doc_id]

    doc_types = None
    sources_filter = None 

    # 4. กระบวนการค้นหาข้อมูลจากฐานเวกเตอร์ (Vector Retrieval Process)
    docs = []
    raw_docs = []

    try:
        # ดึงข้อมูลแบบเฉพาะเจาะจงเพื่อรักษาขอบเขตเนื้อหา
        raw_docs = search_similar(search_query, k=top_k*3, doc_ids=sanitized_doc_ids, sources=sources_filter, doc_types=doc_types)
        logger.info(f"[rag] Found {len(raw_docs)} raw docs")

        # จัดอันดับใหม่ด้วยระบบ AI Scoring
        docs = _rerank_documents(search_query, raw_docs, top_k)
        
        # คัดกรองความมั่นใจเพื่อป้องการนำเสนอเนื้อหาขยะเข้าสู่ LLM
        relevant_docs = _filter_relevant_docs(search_query, docs, min_score=MIN_SCORE_THRESHOLD)
        
        if not relevant_docs:
            qna_match = _find_best_qna_answer_from_docs(search_query, docs) 
            if qna_match:
                return {
                    "answer": qna_match["answer"],
                    "sources": qna_match["sources"],
                    "intent": "qna",
                    "mode": f"{mode}+qna"
                }
            
            return {
                "answer": "ไม่พบข้อมูลที่ตรงกับคำถามในเอกสารที่แนบมาครับ (Relevance Score ต่ำเกินไป)",
                "sources": [],
                "intent": intent,
                "mode": mode
            }
            
        docs = relevant_docs 

    except Exception as e:
        logger.error(f"[rag] Search failed: {e}")
        return {"answer": f"ระบบค้นหาขัดข้อง: {str(e)}", "sources": [], "intent": intent, "mode": mode}

    # 5. การประกอบ Context ย่อย และซ่อนโครงสร้างเพื่อเสถียรภาพในการ Gen (Reference Mapping Logic)
    table_map = {}
    table_cat_map = {}
    context_parts = [] 
    table_counter = 0 
    
    found_table_ids = []
    
    try:
        context_parts.append("⚠️ **แหล่งข้อมูลอ้างอิง:** (เรียงตามความเกี่ยวข้อง)\n")

        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            doc_id = md.get("doc_id", "unknown")
            page = md.get("page", "?")
            source = str(md.get("source", "text")).lower().strip()
            
            content = getattr(d, "page_content", "") or ""
            content = content.replace("\x00", "")
            
            chunk_header = f"[SOURCE {i}] ID: {doc_id} | Page: {page}"

            if source == "table":
                table_counter += 1
                table_ref_id = str(table_counter) 
                found_table_ids.append(table_ref_id)
                
                # ระบบจัดการตารางแบบอ้างอิง: ซ่อนโค้ด HTML/รูปภาพไว้หลังบ้าน แล้วผูกด้วยรหัสเพื่อป้องกัน LLM สร้างโครงสร้างผิดพลาด
                image_path = md.get("image_path")
                if image_path:
                    clean_image_path = str(image_path).replace("\\", "/")
                    img_url = f"/ingested/{doc_id}/{clean_image_path}"
                    table_map[table_ref_id] = f"<img src='{img_url}' alt='Table Data' class='max-w-full h-auto rounded shadow-sm border mx-auto' />"
                else:
                    raw_html = md.get("html_content", "")
                    safe_html = _sanitize_html_content(raw_html)
                    if not safe_html:
                        safe_html = f"<pre class='text-xs overflow-auto p-2 bg-gray-100'>{md.get('markdown_content', 'No content')}</pre>"
                    table_map[table_ref_id] = safe_html
                
                # ควบคุมโครงสร้างการตอบกลับ
                chunk_header += f" | **TYPE: TABLE (Code: [SHOW_TABLE:TBL_{table_ref_id}])**"
                
                category = md.get("category", "").strip().lower()
                role = md.get("role", "").strip().lower()
                
                if category:
                    cat_key = f"cat:{category}"
                    if cat_key not in table_cat_map: table_cat_map[cat_key] = table_map[table_ref_id]
                if role:
                    role_key = f"role:{role}"
                    if role_key not in table_cat_map: table_cat_map[role_key] = table_map[table_ref_id]
            
            context_parts.append(f"{chunk_header}\n{content[:3500]}")

        context_text = "\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"[rag] Context build failed: {e}")
        return {"answer": "เกิดข้อผิดพลาดในการเตรียมข้อมูล", "sources": [], "intent": intent, "mode": mode}
    
    # ------------------------------------------------------------------
    # Prompt Engineering: การควบคุมทิศทางและการทำงานของโมเดล
    # ------------------------------------------------------------------
    if mode == "table":
        # โหมดการทำงาน 1: การสกัดตารางโดยเฉพาะ (Table Extraction Mode)
        system_prompt = (
            "บทบาท: คุณคือระบบ AI อัจฉริยะที่เชี่ยวชาญการสกัดข้อมูลโครงสร้าง (Structured Data Extraction)\n"
            "ภารกิจ: ค้นหา 'ตาราง' ที่ตรงกับคำถามของผู้ใช้มากที่สุดจาก CONTEXT ที่ให้มา\n"
            "\n"
            "ขั้นตอนการทำงาน:\n"
            "1. สแกนหาข้อมูลที่มีระบุว่าเป็น (TYPE: TABLE)\n"
            "2. อ่านหัวข้อตาราง (Table Name/Summary) และเนื้อหาภายในเพื่อตรวจสอบความเกี่ยวข้อง\n"
            "3. การตอบกลับ (Strict Output):\n"
            "   - ถ้าเจอ: ให้ตอบเฉพาะรหัส [SHOW_TABLE:TBL_x] เท่านั้น (ห้ามพูด ห้ามเกริ่นนำ)\n"
            "   - ถ้าเจอหลายตารางที่เกี่ยวข้องกัน: ส่งมาให้ครบ เช่น [SHOW_TABLE:TBL_1] [SHOW_TABLE:TBL_2]\n"
            "   - ถ้าไม่เจอ: ให้ตอบว่า 'NULL'\n"
            "\n"
            f"=== CONTEXT ===\n{context_text}\n==============="
        )
    else:
        # โหมดการทำงาน 2: นักวิเคราะห์ข้อมูลอัจฉริยะแบบผสมผสาน (Smart Analyst)
        system_prompt = (
            "บทบาท: คุณคือ 'ผู้เชี่ยวชาญด้านเอกสาร' ที่เน้นความถูกต้องของข้อมูลสูงสุด\n"
            "หน้าที่: ตอบคำถามจาก Context ที่ให้มา โดยเลือกวิธีนำเสนอที่ดีที่สุด\n"
            "\n"
            "🧠 วิธีการนำเสนอข้อมูล:\n"
            "1. **สำหรับ 'ตารางข้อมูล' หรือ 'แบบฟอร์ม':**\n"
            "   - ให้ใช้ Tag: [SHOW_TABLE:TBL_x] ตามรหัสที่ระบุใน SOURCE เสมอ\n"
            "   - ห้ามวาดตาราง Markdown |...| เองเด็ดขาด!\n"
            "2. **สำหรับ 'รูปภาพทั่วไป':**\n"
            "   - ให้มองหาข้อมูลใน SOURCE ที่มี 'source': 'image' เท่านั้น\n"
            "   - ให้ใช้ Tag: [SHOW_IMAGE: images/ชื่อไฟล์.png] โดยต้องก็อปปี้ค่ามาจาก 'image_path' หรือชื่อไฟล์ใน Metadata มาวางให้ตรงเป๊ะ ห้ามเดาชื่อไฟล์หรือเอาเลขหน้ามาต่อกันเองเด็ดขาด!\n"
            "\n"
            "📋 รูปแบบการตอบ:\n"
            "1. ตอบคำถามให้ตรงประเด็น\n"
            "2. แทรก Tag อ้างอิงประกอบเสมอเมื่ออ้างอิงตารางหรือรูปภาพ\n"
            "3. **[สำคัญมาก]** ห้ามใส่รูปแบบดิบๆ อย่าง [SOURCE 1, SOURCE 2] และ (อ้างอิงจาก SOURCE 3) ท้ายประโยคเด็ดขาด!\n"
            "\n"
            "⚠️ กฎเหล็ก:\n"
            "1. ห้ามพิมพ์ตารางด้วยตัวอักษร ให้ใช้ [SHOW_TABLE:TBL_x] แทนเสมอ\n"
            "2. หากผู้ใช้ถามหา 'บุคคล', 'ชื่อคน' หรือ 'ตัวเลขสำคัญ' ให้ตอบตามข้อมูลใน Context เท่านั้น ห้ามเดาหรือแต่งเรื่องเองเด็ดขาด\n"
            "3. หากค้นหาชื่อบุคคลแล้วไม่พบใน Context ให้ตอบว่า 'ไม่พบข้อมูลที่ระบุชื่อบุคคลนี้'\n"
            "4. หากขอดูรูป แล้วหารูปไม่เจอ ห้ามเดา Path เอง ให้ตอบว่า 'ไม่พบรูปภาพตามที่ขอ'\n"
            "\n"
            f"=== DOCUMENT CONTEXT ===\n{context_text}\n========================"
        )

    # -------------------------------------------------------------------
    # Generation Phase: สถาปัตยกรรมระบบทดแทนอัตโนมัติ (Smart Fallback Architecture)
    # -------------------------------------------------------------------
    llm = _get_llm_instance(model=_LL_MODEL_FAST)
    
    answer_text = ""
    ai_response = None
    
    # ประกอบประวัติและคำถามปัจจุบันเข้าสู่ Message Structure
    messages = [SystemMessage(content=system_prompt)]
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=search_query)) 

    # แผนการทำงานหลัก (Primary Plan): เรียกใช้งาน LLM ตัวหลัก (Qwen)
    try:
        if llm:
            ai_response = await llm.ainvoke(messages)
            answer_text = getattr(ai_response, "content", str(ai_response))
    except Exception as e:
        logger.warning(f"[rag] ❌ Primary LLM failed: {e}")

    # แผนสำรองระดับ 1 (Fallback Layer 1): สลับไปใช้งาน Google Gemini กรณีที่โมเดลหลักขัดข้อง
    if not answer_text or answer_text == "AI Error":
        try:
            google_llm = _get_google_llm()
            if google_llm:
                logger.info("[rag] 🔄 Switching to Backup LLM: Google Gemini...")
                ai_response = await google_llm.ainvoke(messages)
                answer_text = getattr(ai_response, "content", str(ai_response))
        except Exception as e_google:  
            logger.error(f"[rag] ❌ Google LLM also failed: {e_google}")

    # แผนสำรองระดับ 2 (Fallback Layer 2): แสดงข้อมูลดิบ (Raw Fallback) กรณีระบบ AI ขัดข้องทั้งหมด 
    # หรือบังคับแสดงตารางอัตโนมัติเพื่อป้องกันจอดับ (Zero-Downtime Design)
    if not answer_text:
        if intent == "table" and found_table_ids:
             answer_text = f"[SHOW_TABLE:TBL_{found_table_ids[0]}]" 
        else:
             logger.warning("[rag] ⚠️ All LLMs failed. Using Raw Fallback.")
             return {
                 "answer": _generate_fallback_answer(docs, "System Error"), 
                 "sources": [], 
                 "intent": intent, 
                 "mode": f"{mode}+error"
             }
    # -------------------------------------------------------------------
    # Post-Processing Phase: เรนเดอร์ส่วนประกอบ UI ทางฝั่งแบคเอนด์
    # -------------------------------------------------------------------
    if answer_text: 
        try:
            # 1. จัดการตาราง (Priority 1)
            if table_map or table_cat_map:
                pattern_tbl = re.compile(r"\[(?:SHOW_TABLE|SHOW|TABLE)[^:]*:\s*(?:TBL[_]?)?\s*([\d\._]+)\]", re.IGNORECASE)
                def replace_tbl(match):
                    found_id = match.group(1)
                    clean_id = re.sub(r'[^0-9]', '', found_id)
                    raw_html = table_map.get(clean_id) or (list(table_map.values())[0] if len(table_map) == 1 else None)
                    if raw_html:
                        if "<img" in raw_html: return f"\n<div class='my-4'>{raw_html}</div>\n"
                        clean_html = re.sub(r'<table[^>]*>', '<table>', raw_html, flags=re.IGNORECASE)
                        return f"\n<div class='answer-tables-content'>{clean_html}</div>\n"
                    return match.group(0)
                answer_text = pattern_tbl.sub(replace_tbl, answer_text)

            # 2. [จุดชี้ตาย] จัดการรูปภาพ - สแกนหาไฟล์จริงในโฟลเดอร์ images แบบโคตรยืดหยุ่น
            def fix_image_path(match):
                match_val = match.group(1).strip()
                primary_doc_id = docs[0].metadata.get("doc_id", "unknown") if docs else "unknown"
                
                img_dir = INGESTED_DIR / primary_doc_id / "images"
                final_filename = ""
                
                # 1. พยายามแกะเลขหน้าจากสิ่งที่ AI ส่งมา 
                # [อัปเกรด] รองรับ AI หลอนทุกรูปแบบ: page_17, p017, p17, figure_p017_001.png
                page_match = re.search(r'(?:page_?|p)(?:0)*(\d+)', match_val.lower())
                
                if page_match and img_dir.exists():
                    p_num = int(page_match.group(1))
                    p_str = f"p{p_num:03d}" # แปลงให้เป็นมาตรฐาน p017
                    
                    # สแกนหาไฟล์ในโฟลเดอร์จริงๆ
                    for f in img_dir.iterdir():
                        # หาไฟล์ที่มีคำว่า p017 และต้อง "ไม่ใช่" ตาราง (กันมันสับสนกับภาพตาราง)
                        if (p_str in f.name.lower() or f"_p{p_num}_" in f.name.lower()) and "table" not in f.name.lower():
                            final_filename = f.name
                            break

                # 2. ถ้าหาจากเลขหน้าไม่ได้ ลองหาแบบ Exact Match
                if not final_filename:
                    clean_name = match_val.replace("\\", "/").split("/")[-1]
                    if img_dir.exists() and (img_dir / clean_name).exists():
                        final_filename = clean_name
                    else:
                        # 3. Fallback ท่าสุดท้าย: ดึงรูปภาพจริงๆ (Category: figure) รูปแรกใน Context มาโชว์
                        image_metas = [d.metadata for d in docs if d.metadata.get("category") == "figure" or "image" in str(d.metadata.get("source", "")).lower()]
                        if image_metas:
                            final_filename = image_metas[0].get("original_file_name") or image_metas[0].get("image_path", "").split("/")[-1]

                # ถ้ายังหาห่าอะไรไม่เจอจริงๆ
                if not final_filename:
                    final_filename = clean_name if 'clean_name' in locals() else match_val.split("/")[-1]

                # ประกอบ URL 
                final_url = f"/ingested/{primary_doc_id}/images/{final_filename.replace('images/', '')}"
                
                return f"\n<img src='{final_url}' alt='Image Data' class='max-w-full h-auto rounded shadow-sm border mx-auto my-4' onerror=\"this.style.display='none'\" />\n"
            img_pattern = re.compile(r"\[(?:SHOW_IMAGE|IMAGE)\s*:\s*([^\]]+)\]", re.IGNORECASE)
            answer_text = img_pattern.sub(fix_image_path, answer_text)
                
        except Exception as e:
            logger.error(f"[rag] Post-processing failed: {e}")
    # 6) รวบรวมข้อมูลแหล่งอ้างอิงเพื่อสนับสนุนความโปร่งใส (Citation Tracking)
    sources = []
    for d in docs:
        md = d.metadata or {}
        sources.append({
            "doc_id": md.get("doc_id"),
            "page": md.get("page"),
            "source": md.get("source"),
            "chunk_id": md.get("chunk_id")
        })

    return {"answer": answer_text, "sources": sources, "intent": intent, "mode": f"{mode}+qna_llm"}