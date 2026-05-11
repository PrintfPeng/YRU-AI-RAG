# backend/services/query_router.py
from typing import Literal
from langchain_core.prompts import PromptTemplate
from backend.services.llm_provider import LocalLLMProvider

def route_query(query: str) -> Literal["sql", "rag"]:
    """
    วิเคราะห์คำถามของผู้ใช้เพื่อตัดสินใจว่าจะไปทาง SQL Database (เชิงสถิติ/นับจำนวน)
    หรือ Vector Database (เชิงบรรยาย/หาเอกสาร)
    """
    # ใช้ LLM แบบ Temperature=0.0 เพื่อให้ผลลัพธ์คงที่ (Deterministic) ไม่มีความคิดสร้างสรรค์เจือปน
    llm = LocalLLMProvider.get_primary_llm(temperature=0.0)
    
    prompt_template = """คุณคือ AI ระบบผู้เชี่ยวชาญด้านการจำแนกเจตนาคำถาม (Query Routing)
หน้าที่ของคุณคือการพิจารณาว่าคำถามของผู้ใช้ควรถูกส่งไปที่ระบบใดระหว่าง:

1. "sql": สำหรับคำถามที่ต้องการการคำนวณ, สถิติ, การนับจำนวน, หาค่าผลรวม, ค้นหาเจาะจง หรือข้อมูลที่อยู่ในรูปแบบตารางโครงสร้างชัดเจน (เช่น โปรเจกต์มีกี่อัน?, ใครได้งบเยอะสุด?, โครงการ X รหัสอะไร?)
2. "rag": สำหรับคำถามที่ต้องการคำอธิบาย, สรุปเนื้อหา, นโยบาย, หรือรายละเอียดเชิงบรรยาย (เช่น โปรเจกต์นี้มีเป้าหมายหลักเกี่ยวกับอะไร?, อธิบายยุทธศาสตร์ที่ 1 หน่อย)

คำถามของผู้ใช้: "{query}"

จงตอบด้วยคำว่า "sql" หรือ "rag" เท่านั้น ห้ามมีคำอธิบายเพิ่มเติมหรือเครื่องหมายวรรคตอนใดๆ
คำตอบ:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    try:
        print(f"[Router] กำลังวิเคราะห์เจตนาของคำถาม: '{query}'")
        response = chain.invoke({"query": query})
        result = response.content.strip().lower()
        
        # ตัดตัวอักษรขยะทิ้งเผื่อ AI เผลอตอบมา
        if "sql" in result:
            print("[Router] 🔀 ตัดสินใจเลือกเส้นทาง: [SQL Agent]")
            return "sql"
        
        print("[Router] 🔀 ตัดสินใจเลือกเส้นทาง: [Vector/RAG]")
        return "rag"
    except Exception as e:
        print(f"[Router] ⚠️ Error during routing (fallback to RAG): {e}")
        return "rag"
