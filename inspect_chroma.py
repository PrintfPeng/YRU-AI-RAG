# inspect_chroma.py
import sys
import os
from pathlib import Path

# เซ็ต Path ให้เข้าถึงโฟลเดอร์ backend ได้
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from backend.services.vector_store import get_collection_info, get_vector_store

def inspect_database():
    print("🔍 กำลังตรวจสอบสมองของ AI (ChromaDB)...")
    
    # 1. ดูภาพรวมว่ามีกี่รายการแล้ว
    info = get_collection_info()
    if "error" in info:
        print(f"❌ ไม่สามารถอ่านข้อมูลได้: {info['error']}")
        return
        
    print("\n📊 --- สถิติภาพรวม ---")
    print(f"- จำนวนข้อมูล (Sample Count): {info.get('sample_count', 0)}")
    print(f"- แหล่งที่มา (Sources): {info.get('unique_sources', [])}")
    
    # 2. ดูข้อมูลแบบลึกขึ้น
    print("\n📝 --- กำลังดึงตัวอย่างข้อมูลใน Database 3 รายการล่าสุด ---")
    vectordb = get_vector_store()
    
    # ใช้ฟังก์ชัน GET ของ ChromaDB เพื่อดึงข้อมูลดิบออกมาดู
    collection = vectordb._collection
    raw_data = collection.get(limit=3)
    
    if not raw_data or not raw_data['documents']:
        print("📭 ยังไม่มีข้อมูลในระบบเลยครับ")
        return
        
    for i in range(len(raw_data['documents'])):
        print(f"\n[{i+1}] ID: {raw_data['ids'][i]}")
        print(f"📌 Metadata (ข้อมูลกำกับ): {raw_data['metadatas'][i]}")
        print(f"📖 ข้อความ (Text): {raw_data['documents'][i][:200]}... (ตัดมาแค่ 200 ตัวอักษร)")

if __name__ == "__main__":
    inspect_database()
