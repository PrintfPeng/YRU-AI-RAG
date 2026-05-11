# test_db.py
import mysql.connector
import os
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env ที่อยู่ในโฟลเดอร์ Data_Ingestion_Hybrid_RAG
from pathlib import Path
env_path = Path(__file__).parent / "Data_Ingestion_Hybrid_RAG" / ".env"
load_dotenv(dotenv_path=env_path)

def test_connection():
    print("🔄 กำลังเชื่อมต่อกับ MySQL...")
    try:
        # เชื่อมต่อฐานข้อมูลโดยดึงค่าให้ตรงกับใน .env
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "10.10.2.154"),
            user=os.getenv("DB_USER", "ai-sandbox-read"),
            password=os.getenv("DB_PASSWORD", "9IKAjm.R7Qzm_OIZ"), # ใส่รหัสผ่านจริงสำรองไว้เผื่อหา .env ไม่เจอ
            database=os.getenv("DB_NAME", "ai-sandbox_db"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        cursor = conn.cursor(dictionary=True)
        
        # รันคำสั่ง SQL ที่คุณต้องการ
        print("🔄 กำลังดึงข้อมูล...")
        
        # หมายเหตุ: ลองเทสต์ด้วย SHOW TABLES ก่อนเพื่อเช็คว่าต่อ DB ติดไหม
        cursor.execute("SHOW TABLES;") 
        
        # ถ้าอยากดึงข้อมูลโปรเจกต์ (แก้ชื่อ table ให้ตรงกับใน DB จริงของคุณ)
        # cursor.execute("SELECT id, project_name, strategic_plan, description FROM projects LIMIT 5;")
        
        rows = cursor.fetchall()
        
        print(f"✅ ดึงข้อมูลสำเร็จ! พบ {len(rows)} รายการ")
        for row in rows:
            print(row)
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            print("🔒 ปิดการเชื่อมต่อ Database แล้ว")

if __name__ == "__main__":
    test_connection()