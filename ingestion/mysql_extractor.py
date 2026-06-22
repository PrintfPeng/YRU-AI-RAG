import sys
import os
import gc
from pathlib import Path

# ปรับแก้ Path เพื่อให้ Python มองเห็นโฟลเดอร์ backend ที่อยู่ด้านนอก
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import mysql.connector
from langchain_core.documents import Document
from backend.services.vector_store import get_vector_store
# สมมติว่าคุณมีฟังก์ชันตัดคำจาก Ingestion Pipeline เดิม
from backend.services.chunking import preprocess_thai 

import os
from dotenv import load_dotenv

load_dotenv()

def extract_all_tables():
    """
    ดึงข้อมูลจาก 'ทุกตาราง' ใน MySQL โดยอัตโนมัติ 
    แปลงข้อมูลแต่ละบรรทัด (Row) เป็นข้อความ (Text) แล้วทำ Embedding ลง ChromaDB
    """
    print("🔄 Connecting to MySQL...")
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        cursor = conn.cursor(dictionary=True)
        
        # 1. ค้นหาตารางทั้งหมดใน Database
        cursor.execute("SHOW TABLES;")
        tables_result = cursor.fetchall()
        
        # ดึงชื่อตารางออกมาเป็น List (คำสั่ง SHOW TABLES จะคืนค่า key เป็นชื่อ database)
        db_name = os.getenv("DB_NAME", "ai-sandbox_db")
        table_key = f"Tables_in_{db_name}"
        
        # รองรับกรณีที่ key ไม่ตรงกับ Tables_in_...
        if tables_result and table_key not in tables_result[0]:
            table_key = list(tables_result[0].keys())[0]

        tables = [row[table_key] for row in tables_result]
        print(f"📂 พบตารางทั้งหมด {len(tables)} ตาราง: {tables}")
        
        documents = []
        
        # 2. วนลูปดึงข้อมูลทีละตาราง
        for table in tables:
            print(f"⏳ กำลังดึงข้อมูลจากตาราง: {table} ...")
            
            # ป้องกัน Connection Timeout ระหว่างดึงตารางที่ใช้เวลานาน
            try:
                conn.ping(reconnect=True, attempts=3, delay=2)
            except Exception:
                pass
                
            # จำกัดการดึงข้อมูลตารางละ 100 แถว เพื่อป้องกัน Memory เต็มช่วงทดสอบ
            # (ถ้าต้องการดึงทั้งหมด ให้เอา LIMIT 100 ออก)
            cursor.execute(f"SELECT * FROM `{table}` LIMIT 100;")
            rows = cursor.fetchall()
            
            for row in rows:
                # 3. Dynamic Textualization (แปลง Row เป็นประโยคภาษาคนอัตโนมัติ)
                row_text_parts = [f"ข้อมูลจากหมวดหมู่ (ตาราง) {table}:"]
                
                # วนลูปเอาชื่อคอลัมน์และข้อมูลมาต่อกัน
                for col_name, value in row.items():
                    # ข้ามคอลัมน์ที่ข้อมูลว่างเปล่า
                    if value is not None and str(value).strip() != "":
                        row_text_parts.append(f" {col_name} คือ {value}")
                        
                raw_text = ",".join(row_text_parts)
                
                # 4. Thai NLP Support: ตัดคำด้วย PyThaiNLP
                clean_text = preprocess_thai(raw_text)
                
                # 5. สร้าง Document Object พร้อม Metadata อ้างอิงว่ามาจากตารางไหน
                # หากมีคอลัมน์ id ให้นำมาใช้เป็น reference 
                row_id = row.get("id", "unknown")
                doc = Document(
                    page_content=clean_text,
                    metadata={"source": "mysql", "table": table, "row_id": str(row_id)}
                )
                documents.append(doc)

        print(f"✅ ดึงข้อมูลรวมทั้งหมด {len(documents)} รายการจากทุกตาราง")
        
        # 6. นำเข้า ChromaDB
        if documents:
            vector_store = get_vector_store()
            print("🔄 Ingesting into ChromaDB... (กระบวนการนี้อาจใช้เวลานานหากข้อมูลเยอะ)")
            
            # แบ่งการบันทึกข้อมูลออกเป็นชุดๆ (Batching) ชุดละ 5000 รายการ
            # เพื่อป้องกัน Error: Batch size is greater than max batch size of 5461 ของ ChromaDB
            batch_size = 5000
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                current_batch = (i // batch_size) + 1
                print(f"📦 กำลังบันทึก Batch ที่ {current_batch}/{total_batches} (จำนวน {len(batch)} รายการ)...")
                vector_store.add_documents(batch)
                
            print("✅ Ingestion Complete! บันทึกลงสมอง AI เรียบร้อยแล้ว")
        else:
            print("⚠️ ไม่พบข้อมูลใดๆ ในตารางเลย")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
        # ChromaDB Production Stability: ป้องกัน File Lock
        gc.collect()

if __name__ == "__main__":
    extract_all_tables()
