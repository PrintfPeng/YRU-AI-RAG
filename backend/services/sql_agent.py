# backend/services/sql_agent.py
import os
import re
import traceback
import mysql.connector
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from backend.services.llm_provider import LocalLLMProvider

# 1. โหลด Environment Variables ทันทีที่ไฟล์ถูกเรียกใช้
load_dotenv()

def get_db_connection():
    """ศูนย์รวมการเชื่อมต่อฐานข้อมูล ลดการเขียนโค้ดซ้ำซ้อน"""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "10.10.2.154"),
        user=os.getenv("DB_USER", "ai-sandbox-read"),
        password=os.getenv("DB_PASSWORD", "9IKAjm.R7Qzm_OIZ"),
        database=os.getenv("DB_NAME", "ai-sandbox_db"),
        port=int(os.getenv("DB_PORT", 3306))
    )

def get_db_schema() -> str:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SHOW TABLES;")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_text = ""
        for table in tables:
            cursor.execute(f"DESCRIBE `{table}`;")
            columns = cursor.fetchall()
            col_details = [f"{col[0]} ({col[1]})" for col in columns]
            schema_text += f"Table: {table}\nColumns: {', '.join(col_details)}\n\n"
            
        return schema_text
    except Exception as e:
        print(f"[SQL_Agent] ❌ Error fetching schema: {e}")
        return ""
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def clean_and_validate_sql(raw_text: str) -> str:
    """
    ฟังก์ชันสกัดและทำความสะอาด SQL จาก LLM อย่างชาญฉลาดและปลอดภัย
    """
    # 2. ค้นหาเฉพาะข้อมูลที่อยู่ใน Markdown Block (หาก AI ใส่มา)
    code_block_match = re.search(r'```(?:sql|SQL)?\s*(.*?)\s*```', raw_text, flags=re.DOTALL)
    if code_block_match:
        sql = code_block_match.group(1).strip()
    else:
        sql = raw_text.strip()
        
    # 3. สลัดข้อความเกริ่นนำทิ้ง โดยมองหาคำว่า SELECT ตัวแรก
    select_match = re.search(r'(?i)\bSELECT\b[\s\S]*', sql)
    if not select_match:
        raise ValueError("INVALID_SQL_TYPE")
        
    final_sql = select_match.group(0).strip()
    
    # 4. ตรวจสอบความปลอดภัยสูงสุด: คำสั่งจะต้องขึ้นต้นด้วย SELECT เท่านั้น
    # วิธีนี้จะแก้ปัญหา False Positive จากคำว่า updated_at หรือ drop ได้ 100%
    if not final_sql.upper().startswith("SELECT"):
        raise ValueError("INVALID_SQL_TYPE")
        
    return final_sql

def generate_and_run_sql(query: str) -> str:
    print("[SQL_Agent] 🔍 เริ่มต้นกระบวนการ Text-to-SQL...")
    
    schema = get_db_schema()
    if not schema:
        return "ขออภัยครับ ไม่สามารถอ่านโครงสร้างฐานข้อมูลได้ในขณะนี้"
        
    try:
        llm = LocalLLMProvider.get_primary_llm(temperature=0.0)
    except Exception as e:
        print(f"[SQL_Agent] ❌ Error initializing LLM: {e}")
        return "ขออภัยครับ ไม่สามารถเชื่อมต่อกับโมเดล AI ได้"
    
    sql_prompt = PromptTemplate.from_template("""คุณคือ Data Analyst ที่เชี่ยวชาญภาษา MySQL
นี่คือโครงสร้างตารางทั้งหมดในฐานข้อมูล:
{schema}

คำถามจากผู้ใช้: "{query}"

จงเขียนคำสั่ง SQL ที่ถูกต้องเพื่อหาคำตอบสำหรับคำถามนี้
ข้อบังคับ:
- ห้ามใช้คำสั่งที่เป็นอันตราย (INSERT, UPDATE, DELETE, DROP) ใช้ได้แค่ SELECT เท่านั้น
- ตอบกลับมาเฉพาะคำสั่ง SQL เท่านั้น ห้ามอธิบาย ห้ามใส่ ```sql นำหน้าหรือตามหลัง
""")
    
    sql_chain = sql_prompt | llm
    
    print("[SQL_Agent] 🧠 กำลังวิเคราะห์และเขียนคำสั่ง SQL...")
    
    try:
        sql_query_response = sql_chain.invoke({"schema": schema, "query": query})
        raw_sql = sql_query_response.content
        print(f"[SQL_Agent] 💻 ผลลัพธ์ดิบจาก AI: \n{raw_sql}")
        
        # ใช้งานกระบวนการสกัดและตรวจสอบที่เขียนใหม่
        safe_sql = clean_and_validate_sql(raw_sql)
        print(f"[SQL_Agent] ✅ คำสั่ง SQL ที่ผ่านการตรวจสอบแล้ว: \n{safe_sql}")
        
    except ValueError as ve:
        if str(ve) == "INVALID_SQL_TYPE":
            print("[SQL_Agent] ⚠️ ตรวจพบคำสั่งที่ไม่ใช่ SELECT หรือมีความเสี่ยง")
            return "ขออภัยครับ เพื่อความปลอดภัย ระบบอนุญาตให้สร้างคำสั่งค้นหาข้อมูล (SELECT) เท่านั้น"
        return "ขออภัยครับ ไม่สามารถตีความคำสั่งที่สร้างขึ้นได้"
    except Exception as e:
        print(f"[SQL_Agent] ❌ เกิดข้อผิดพลาดในขั้นตอนสร้าง SQL: {e}")
        return "ขออภัยครับ ระบบสร้างคำสั่งค้นหาข้อมูลล้มเหลว"
    
    conn = None
    try:
        conn = get_db_connection()
        # dictionary=True เพื่อให้อ่านข้อมูลง่ายเมื่อส่งไปให้ LLM สรุป
        cursor = conn.cursor(dictionary=True) 
        
        # MySQL Connector มีระบบป้องกัน Multi-Statements โดยค่าเริ่มต้น
        # หากมีใครพยายามต่อท้ายด้วย ; DROP TABLE ... ตัว Connector จะโยน Error ให้ทันที
        cursor.execute(safe_sql)
        results = cursor.fetchall()
        
        print(f"[SQL_Agent] 📊 รัน SQL สำเร็จ ได้ผลลัพธ์มา {len(results)} รายการ")
        
        if not results:
            return "ไม่พบข้อมูลที่ตรงกับคำถามของคุณในฐานข้อมูลครับ"
            
        answer_prompt = PromptTemplate.from_template("""คุณคือผู้ช่วย AI ที่ชาญฉลาดและตอบคำถามได้อย่างเป็นธรรมชาติ
คำถามจากผู้ใช้: "{query}"
ข้อมูลที่ดึงมาจากฐานข้อมูลเพื่อตอบคำถาม:
{results}

จงสรุปและตอบคำถามผู้ใช้ด้วยภาษาไทยที่อ่านง่าย อ้างอิงจากข้อมูลด้านบนเท่านั้น
หากข้อมูลมีหลายรายการ ให้สรุปเป็นข้อๆ หรือภาพรวมให้เข้าใจง่าย
""")
        answer_chain = answer_prompt | llm
        
        print("[SQL_Agent] 🗣️ กำลังสรุปข้อมูลเป็นภาษาคน...")
        final_response = answer_chain.invoke({
            "query": query, 
            "results": str(results[:50]) # ตัดแค่ 50 รายการ ป้องกัน Context Window (Token) ล้น
        })
        
        return final_response.content.strip()
        
    except mysql.connector.Error as err:
        # 5. แยกแยะ Error Code และคืนข้อความที่มีประโยชน์เฉพาะจุด
        print(f"[SQL_Agent] ❌ MySQL Error [{err.errno}]: {err.msg}")
        
        if err.errno == 1146:
            return "ขออภัยครับ ระบบ AI อ้างอิงตารางที่ไม่มีอยู่จริง (รหัส 1146)"
        elif err.errno == 1054:
            return "ขออภัยครับ ระบบ AI อ้างอิงชื่อคอลัมน์ที่ไม่ถูกต้อง (รหัส 1054)"
        elif err.errno == 1064:
            return "ขออภัยครับ คำสั่ง SQL ที่ AI สร้างขึ้นมีข้อผิดพลาดทางไวยากรณ์ (รหัส 1064)"
        elif err.errno in (1044, 1045):
            return "ขออภัยครับ เกิดปัญหาการยืนยันสิทธิ์ในการเข้าถึงฐานข้อมูล"
        else:
            return f"ขออภัยครับ เกิดข้อผิดพลาดจากฐานข้อมูล (รหัสข้อผิดพลาด: {err.errno})"
            
    except Exception as e:
        print(f"[SQL_Agent] ❌ System Error:")
        traceback.print_exc() # พิมพ์ Stack Trace เชิงลึกให้ Developer
        return "ขออภัยครับ ระบบประมวลผลข้อมูลล้มเหลว"
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
