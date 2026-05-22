# scripts/run_mysql_ingestion.py
"""
สคริปต์สำหรับการนำเข้าข้อมูล (Data Ingestion Pipeline)
ดึงข้อมูลโครงการจากระบบแผนงาน/ยุทธศาสตร์ (MySQL) -> จัดรูปแบบข้อความ -> ทำ Embedding -> บันทึกลง ChromaDB
รองรับทั้งโหมด Local (Persistent Disk) และ Remote (Client/Server) ตามค่าในไฟล์ .env
"""

import sys
import os
import gc
import json
import logging
from pathlib import Path
from decimal import Decimal
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
import mysql.connector

# กำหนดระดับการล็อก
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# แก้ปัญหา Encoding บน Windows Terminal ให้สามารถแสดงผลภาษาไทยได้ถูกต้อง
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ดึงตำแหน่ง ROOT ของโปรเจกต์
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# โหลด Environment Variables
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def get_db_connection() -> mysql.connector.MySQLConnection:
    """เชื่อมต่อกับ MySQL Database อย่างปลอดภัย"""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "10.10.2.154"),
            user=os.getenv("DB_USER", "ai-sandbox-read"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "ai-sandbox_db"),
            port=int(os.getenv("DB_PORT", 3306)),
            charset="utf8mb4"  # หรือ utf8
        )
        return conn
    except Exception as e:
        logger.error(f"❌ ไม่สามารถเชื่อมต่อฐานข้อมูล MySQL ได้: {e}")
        raise e


def get_embedding_model():
    """
    เตรียมโมเดลสำหรับแปลงข้อความเป็นเวกเตอร์ (Embedding Model)
    พยายามใช้โมเดล bge-m3 ผ่านระบบ Remote (Open WebUI/Ollama) ก่อน
    หากไม่มีจะสลับมาใช้ sentence-transformers ในเครื่องแบบ Local อัตโนมัติ (Fallback)
    """
    base_url = os.getenv("OPEN_WEBUI_BASE_URL")
    api_key = os.getenv("OPEN_WEBUI_API_KEY", "ollama")
    
    # วิธีที่ 1: ดึงผ่าน Open WebUI/Ollama API
    if base_url:
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info(f"⏳ กำลังเปิดใช้งาน Embedding Model (bge-m3:latest) ผ่าน API: {base_url} ...")
            return OpenAIEmbeddings(
                model="bge-m3:latest",
                base_url=base_url,
                api_key=api_key,
                check_embedding_ctx_length=False
            )
        except Exception as e:
            logger.warning(f"⚠️ ไม่สามารถเชื่อมต่อ API Embedding ได้: {e}. สลับไปใช้ Local Model...")

    # วิธีที่ 2: รันในเครื่องเครื่องแบบ Local (Fallback)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("⏳ กำลังโหลด Local Embedding Model (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) เข้าสู่หน่วยความจำ...")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logger.error(f"❌ ไม่สามารถเริ่มต้นโหลด Embedding Model ได้: {e}")
        raise e


def get_chromadb_client(collection_name: str = "yru_planning_data") -> Tuple[Any, Any]:
    """
    สร้าง ChromaDB Client และดึง Collection
    รองรับ:
    1. Client/Server Mode (เมื่อมี CHROMA_SERVER_HOST ใน .env)
    2. Persistent Local Mode (เมื่อไม่กำหนดค่า Host)
    """
    import chromadb
    from langchain_chroma import Chroma
    
    chroma_host = os.getenv("CHROMA_SERVER_HOST")
    chroma_port = os.getenv("CHROMA_SERVER_PORT", "8000")
    embeddings = get_embedding_model()

    if chroma_host:
        logger.info(f"🌐 กำลังเชื่อมต่อไปยังเครื่องเซิร์ฟเวอร์ ChromaDB: {chroma_host}:{chroma_port} ...")
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
    else:
        persist_dir = PROJECT_ROOT / "chroma_db"
        logger.info(f"📂 กำลังเปิดใช้งาน ChromaDB แบบ Local ที่โฟลเดอร์: {persist_dir} ...")
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(path=str(persist_dir))
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir)
        )
        
    return client, vector_store


def format_project_to_text(row: Dict[str, Any]) -> str:
    """
    สกัดและจัดรูปแบบข้อมูลแถวโครงการ (Row) 
    ให้ออกมาเป็นบทความสั้น/ข้อความอธิบายแบบประโยคที่มนุษย์เข้าใจง่าย (Textualization)
    """
    project_id = row.get("id")
    project_name = row.get("project_name") or "ไม่ระบุชื่อโครงการ"
    year = row.get("year") or "ไม่ระบุปี"
    department = row.get("department_name") or "ไม่ระบุหน่วยงาน"
    strategic = row.get("strategic_name") or "ไม่ระบุยุทธศาสตร์"
    plan = row.get("plan_name") or "ไม่ระบุแผนงาน"
    
    budget_val = row.get("total_budget") or 0.0
    # แปลง Decimal/Float เป็นเงินบาทรูปแบบอ่านง่าย
    budget_str = f"{float(budget_val):,.2f} บาท" if budget_val else "0.00 บาท"

    principle = str(row.get("principle") or "").strip()
    objective = str(row.get("objective") or "").strip()
    expect = str(row.get("expect") or "").strip()

    # สร้างข้อมูลแบบมีโครงสร้าง (Structured Description)
    text_parts = [
        f"โครงการหลัก: {project_name}",
        f"รหัสโครงการ: {project_id}",
        f"ปีงบประมาณ พ.ศ.: {year}",
        f"หน่วยงานที่รับผิดชอบ: {department}",
        f"ยุทธศาสตร์สนับสนุน: {strategic}",
        f"แผนงานหลัก: {plan}",
        f"งบประมาณทั้งหมด: {budget_str}"
    ]

    if principle:
        # ลบช่องว่างส่วนเกินและรักษารูปแบบ
        clean_principle = " ".join(principle.split())
        text_parts.append(f"หลักการและเหตุผล: {clean_principle}")
        
    if objective:
        clean_objective = " ".join(objective.split())
        text_parts.append(f"วัตถุประสงค์ของโครงการ: {clean_objective}")
        
    if expect:
        clean_expect = " ".join(expect.split())
        text_parts.append(f"ผลลัพธ์ที่คาดว่าจะได้รับ: {clean_expect}")

    return "\n".join(text_parts)


def clean_thai_text(text: str) -> str:
    """
    ทำความสะอาดข้อความ และประมวลผลคำศัพท์ภาษาไทยเบื้องต้น
    ตัดคำและเว้นวรรคให้เหมาะสมเพื่อให้เวกเตอร์แม่นยำขึ้น
    """
    try:
        from pythainlp import word_tokenize
        # จัดการช่องว่างส่วนเกิน
        text = " ".join(text.split())
        # ตัดคำด้วยเอนจินภาษาไทยกลาง
        words = word_tokenize(text, engine='newmm', keep_whitespace=False)
        return " ".join(words)
    except Exception:
        # หากไม่พบ pythainlp หรือเกิดข้อผิดพลาด ให้คืนค่าข้อความปกติ
        return text


def fetch_and_format_projects() -> List[Dict[str, Any]]:
    """ดึงข้อมูลโครงการ ยุทธศาสตร์ หน่วยงาน และงบประมาณมาประกอบร่างและจัดรูปแบบ"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT 
        p.id, 
        py.name AS project_name, 
        py.year, 
        d.name AS department_name, 
        s.name AS strategic_name, 
        pl.name AS plan_name,
        p.principle, 
        p.objective, 
        p.expect, 
        (COALESCE(p.budget1, 0) + COALESCE(p.budget2, 0) + COALESCE(p.budget3, 0) + COALESCE(p.budget4, 0)) AS total_budget
    FROM projects p
    LEFT JOIN project_template_years py ON p.project_template_year_id = py.id
    LEFT JOIN departments d ON p.department_id = d.id
    LEFT JOIN strategics s ON p.strategic_id = s.id
    LEFT JOIN plans pl ON p.plan_id = pl.id
    """
    
    logger.info("⏳ กำลังเริ่มคิวรีข้อมูลโครงการจาก MySQL...")
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        logger.info(f"✅ ดึงข้อมูลสำเร็จ! พบรายการข้อมูลรวม {len(rows)} โครงการ")
        
        documents = []
        for index, row in enumerate(rows):
            # จัดรูปแบบให้อยู่ในโครงสร้างข้อความ
            raw_content = format_project_to_text(row)
            # ปรับแต่งคำภาษาไทยให้อ่านง่ายสำหรับ AI
            clean_content = clean_thai_text(raw_content)
            
            # บันทึกข้อมูลพร้อม Metadata
            doc_data = {
                "text": clean_content,
                "metadata": {
                    "source": "mysql_planning",
                    "table": "projects",
                    "project_id": int(row["id"]),
                    "year": int(row["year"]) if row["year"] else 0,
                    "department": row["department_name"] or "ไม่ระบุ",
                    "budget": float(row["total_budget"]) if row["total_budget"] else 0.0
                },
                "id": f"project_{row['id']}"
            }
            documents.append(doc_data)
            
            if (index + 1) % 100 == 0 or (index + 1) == len(rows):
                logger.info(f"📂 ดำเนินการจัดรูปแบบข้อมูลแล้ว: {index + 1}/{len(rows)}")
                
        return documents
    finally:
        cursor.close()
        conn.close()


def run_ingestion():
    logger.info("🚀 เริ่มต้นกระบวนการ MySQL Data Ingestion Pipeline...")
    
    # 1. ดึงข้อมูลและแปลงเป็นข้อความ (Data Transformation)
    try:
        documents = fetch_and_format_projects()
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในขั้นตอนคิวรีและแปลงข้อมูล: {e}")
        return
        
    if not documents:
        logger.warning("⚠️ ไม่พบข้อมูลที่จะนำเข้า")
        return
        
    # 2. เชื่อมต่อไปยังฐานข้อมูลเวกเตอร์ ChromaDB
    try:
        client, vector_store = get_chromadb_client(collection_name="yru_planning_data")
    except Exception as e:
        logger.error(f"❌ ไม่สามารถเชื่อมต่อกับ ChromaDB ได้: {e}")
        return

    # 3. ส่งข้อมูลไปทำ Embedding และเก็บลง ChromaDB แบบ Batch (ป้องกัน OOM และลดภาระ API)
    batch_size = 100  # ปรับแต่งได้ตามขนาดแรมและ API Rate Limit
    total_docs = len(documents)
    
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    ids = [doc["id"] for doc in documents]

    logger.info(f"📦 กำลังดำเนินการบันทึกข้อมูลเข้า ChromaDB ทั้งหมด {total_docs} รายการ...")
    
    try:
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            batch_texts = texts[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            logger.info(f"⏳ กำลังบันทึกชุดข้อมูล (Batch) ที่ {i//batch_size + 1} ... (รายการที่ {i} ถึง {end_idx - 1})")
            
            # บันทึกลง ChromaDB
            vector_store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            # เคลียร์แคชและเก็บกวาดหน่วยความจำป้องกันการรั่วไหล
            gc.collect()
            
        logger.info("🎉 กระบวนการนำเข้าข้อมูลเสร็จสมบูรณ์เรียบร้อยแล้ว!")
        
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดระหว่างส่งข้อมูลเข้า ChromaDB: {e}")


if __name__ == "__main__":
    run_ingestion()
