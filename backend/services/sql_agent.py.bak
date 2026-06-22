# backend/services/sql_agent.py
import os
import mysql.connector
from langchain_core.prompts import PromptTemplate
from backend.services.llm_provider import LocalLLMProvider

def get_db_schema() -> str:
    """
    อ่านโครงสร้างตาราง (Schema) ทั้งหมดใน Database เพื่อเป็นบริบทให้ LLM ใช้เขียนโค้ด SQL
    """
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "10.10.2.154"),
            user=os.getenv("DB_USER", "ai-sandbox-read"),
            password=os.getenv("DB_PASSWORD", "9IKAjm.R7Qzm_OIZ"),
            database=os.getenv("DB_NAME", "ai-sandbox_db"),
            port=int(os.getenv("DB_PORT", 3306))
        )
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
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def generate_and_run_sql(query: str) -> str:
    """
    รับคำถาม -> สั่ง LLM เขียน SQL -> รันบน MySQL -> ส่งผลลัพธ์ดิบให้ LLM สรุปเป็นภาษาคน
    """
    print("[SQL_Agent] 🔍 เริ่มต้นกระบวนการ Text-to-SQL...")
    
    schema = get_db_schema()
    if not schema:
        return "ขออภัยครับ ไม่สามารถอ่านโครงสร้างฐานข้อมูลได้ในขณะนี้"
        
    llm = LocalLLMProvider.get_primary_llm(temperature=0.0)
    
    # 1. ให้ LLM แปลงคำถามเป็นคำสั่ง SQL
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
    sql_query_response = sql_chain.invoke({"schema": schema, "query": query})
    
    # ทำความสะอาดโค้ด SQL ที่ได้มาจาก LLM
    raw_sql = sql_query_response.content.strip()
    raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    
    print(f"[SQL_Agent] 💻 คำสั่ง SQL ที่ได้: \n{raw_sql}")
    
    # ตรวจสอบความปลอดภัยเบื้องต้น (Safety Check)
    if any(keyword in raw_sql.upper() for keyword in ["DROP", "DELETE", "UPDATE", "INSERT"]):
        return "ขออภัยครับ คำสั่ง SQL นี้มีความเสี่ยงด้านความปลอดภัย ระบบจึงไม่อนุญาตให้ทำงาน"
    
    # 2. รันคำสั่ง SQL
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "10.10.2.154"),
            user=os.getenv("DB_USER", "ai-sandbox-read"),
            password=os.getenv("DB_PASSWORD", "9IKAjm.R7Qzm_OIZ"),
            database=os.getenv("DB_NAME", "ai-sandbox_db"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(raw_sql)
        results = cursor.fetchall()
        
        print(f"[SQL_Agent] 📊 รัน SQL สำเร็จ ได้ผลลัพธ์มา {len(results)} รายการ")
        
        if not results:
            return "ไม่พบข้อมูลที่ตรงกับคำถามของคุณในฐานข้อมูลครับ"
            
        # 3. ให้ LLM แปลงผลลัพธ์ SQL เป็นคำตอบภาษาคน
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
            # ตัดข้อมูลให้เหลือไม่เกิน 50 แถว ป้องกัน Context ล้น (Token Limit)
            "results": str(results[:50]) 
        })
        
        return final_response.content.strip()
        
    except mysql.connector.Error as err:
        error_msg = f"เกิดข้อผิดพลาดจาก MySQL: {err}"
        print(f"[SQL_Agent] ❌ {error_msg}")
        return "ขออภัยครับ คำสั่ง SQL ที่สร้างขึ้นมีปัญหา ไม่สามารถดึงข้อมูลได้"
    except Exception as e:
        error_msg = f"เกิดข้อผิดพลาดในการรันระบบ SQL: {e}"
        print(f"[SQL_Agent] ❌ {error_msg}")
        return "ขออภัยครับ ระบบประมวลผลข้อมูลล้มเหลว"
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
