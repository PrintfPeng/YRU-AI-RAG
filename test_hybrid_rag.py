# test_hybrid_rag.py
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# เซ็ต Path ให้ดึงไฟล์จากโฟลเดอร์ backend ได้
project_root = Path(__file__).resolve().parent / "Data_Ingestion_Hybrid_RAG"
sys.path.append(str(project_root))
load_dotenv(dotenv_path=project_root / ".env")

from backend.services.query_router import route_query
from backend.services.sql_agent import generate_and_run_sql
from backend.services.rag import answer_question

def main():
    print("🤖 ยินดีต้อนรับสู่ Hybrid RAG System (กด Ctrl+C เพื่อออก)")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n📝 พิมพ์คำถามของคุณ: ")
            if not query.strip():
                continue
                
            print("\n" + "="*50)
            # 1. ให้ Router วิเคราะห์เจตนา
            route_decision = route_query(query)
            
            # 2. แยกสายการทำงานตามที่ Router ตัดสินใจ
            if route_decision == "sql":
                answer = generate_and_run_sql(query)
            else:
                print("[RAG] 🔍 เริ่มต้นกระบวนการค้นหาเอกสารจาก ChromaDB...")
                # สมมติว่าใน rag.py มีฟังก์ชัน answer_question(query) ให้เรียกใช้
                # ถ้าฟังก์ชันชื่ออื่น สามารถแก้ตรงนี้ได้เลยครับ
                try:
                    answer = answer_question(query, chat_history=[])
                except Exception as e:
                    answer = f"Error in RAG: {e}"
                    
            print("\n💡 คำตอบที่ได้:")
            print(answer)
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n👋 ลาก่อนครับ!")
            break
        except Exception as e:
            print(f"\n❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()
