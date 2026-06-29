"""
backend/services/smart_router.py
จำแนกคำถามและ route ไปยัง Bot ที่เหมาะสม:
  - Student Bot  : Dify API (STUDENT_BOT_API_KEY) -- คำถามเกี่ยวกับนักศึกษา
  - Planning Bot : Local RAG (answer_question)     -- คำถามเกี่ยวกับกองแผน/YRU
"""
import json
import os
from typing import Literal

import asyncio
import requests
import urllib3
from backend.services.llm_provider import LocalLLMProvider

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DIFY_CHAT_URL = "https://122.154.50.92/external/api/chat-messages"
STUDENT_BOT_API_KEY: str = os.getenv("STUDENT_BOT_API_KEY", "")

RouteType = Literal["student", "planning"]

_STUDENT_KEYWORDS: set = {
    "นักศึกษา", "นศ.", "นักเรียน", "รหัสนักศึกษา", "บัตรนักศึกษา", "ชั้นปี",
    "ลงทะเบียน", "ถอนวิชา", "เพิ่มวิชา", "หน่วยกิต", "รายวิชา",
    "ผลการเรียน", "เกรด", "เกรดเฉลี่ย", "gpa", "gpax", "transcript",
    "ใบรับรอง", "ใบแสดงผล",
    "ตารางเรียน", "ตารางสอบ", "สอบไล่", "สอบกลางภาค", "ปลายภาค",
    "ค่าเทอม", "ค่าธรรมเนียม", "ค่าหน่วยกิต",
    "ทุนการศึกษา", "กยศ", "กรอ",
    "หลักสูตร", "สาขาวิชา",
    "ฝึกประสบการณ์", "สหกิจ", "ฝึกงาน",
    "วิทยานิพนธ์", "สารนิพนธ์",
    "พักการเรียน", "ลาออก", "โอนย้าย",
    "จบการศึกษา", "สำเร็จการศึกษา",
    "yru-tep", "toeic", "toefl",
    "อาจารย์ที่ปรึกษา", "ภาคการศึกษา", "ปีการศึกษา",
}

_PLANNING_KEYWORDS: set = {
    "กองแผน", "ฝ่ายวางแผน", "แผนงาน", "แผนพัฒนา", "แผนปฏิบัติการ", "แผนกลยุทธ์",
    "งบประมาณ", "แผนงบประมาณ",
    "โครงการ",
    "ยุทธศาสตร์", "กลยุทธ์", "วิสัยทัศน์", "พันธกิจ", "เป้าประสงค์",
    "ตัวชี้วัด", "kpi", "okr",
    "รายงานประจำปี", "ผลการดำเนินงาน",
    "ประกันคุณภาพ", "qa", "edpex", "tqa", "iqa", "eqa", "sar",
    "สภามหาวิทยาลัย", "อธิการบดี", "รองอธิการ",
    "นโยบาย", "พระราชบัญญัติ", "ระเบียบมหาวิทยาลัย",
}


def classify_intent_keyword(query: str) -> RouteType:
    text = query.lower()
    student_score = sum(1 for kw in _STUDENT_KEYWORDS if kw in text)
    planning_score = sum(1 for kw in _PLANNING_KEYWORDS if kw in text)
    if student_score > planning_score:
        return "student"
    return "planning"



_FEW_SHOT_PROMPT = """จำแนกคำถามเป็น "student" หรือ "planning"

student  = คำถามของนักศึกษา เกี่ยวกับ เกรด ผลการเรียน ลงทะเบียน ค่าเทอม ทุน กยศ transcript ตารางเรียน ตารางสอบ สาขา ฝึกงาน
planning = คำถามเกี่ยวกับมหาวิทยาลัย/กองแผน เกี่ยวกับ โครงการ งบประมาณ ยุทธศาสตร์ KPI ตัวชี้วัด นโยบาย กองแผน แผนงาน

ตัวอย่าง:
Q: นักศึกษาดูผลการเรียนได้ที่ไหน → student
Q: ตัวชี้วัด KPI ของกองแผน → planning
Q: ค่าเทอมคณะวิทยาศาสตร์เท่าไร → student
Q: งบประมาณประจำปีของ YRU → planning
Q: กยศ กู้ได้วงเงินเท่าไร → student
Q: โครงการยุทธศาสตร์มหาวิทยาลัย → planning

Q: {query} →"""


async def classify_intent_llm(query: str) -> RouteType:
    """ใช้ primary LLM + few-shot prompt จำแนก intent (ทดสอบแล้ว 8/8)"""
    try:
        llm = LocalLLMProvider.get_primary_llm(temperature=0.0)
        prompt = _FEW_SHOT_PROMPT.format(query=query)
        result = await llm.ainvoke(prompt)
        first_word = result.content.strip().split()[0].lower() if result.content.strip() else ""
        route: RouteType = "student" if "student" in first_word else "planning"
        print(f"[SmartRouter] LLM: '{query[:60]}' → {route} (raw: {repr(result.content.strip()[:30])})", flush=True)
        return route
    except Exception as e:
        print(f"[SmartRouter] LLM classify failed ({e}), using keyword fallback", flush=True)
        return classify_intent_keyword(query)


def classify_intent(query: str) -> RouteType:
    """Sync wrapper — keyword scoring (ใช้ใน context ที่ไม่มี event loop)"""
    return classify_intent_keyword(query)


def call_student_bot(query: str, conversation_id: str = "", user_id: str = "public-user") -> dict:
    if not STUDENT_BOT_API_KEY:
        raise ValueError("STUDENT_BOT_API_KEY is not set in .env")
    payload = {
        "query": query,
        "inputs": {},
        "response_mode": "blocking",
        "conversation_id": conversation_id or "",
        "user": user_id,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STUDENT_BOT_API_KEY}",
    }
    resp = requests.post(DIFY_CHAT_URL, json=payload, headers=headers, verify=False, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return {
        "answer": data.get("answer", ""),
        "conversation_id": data.get("conversation_id", ""),
        "message_id": data.get("message_id", ""),
    }


def call_student_bot_stream(query: str, conversation_id: str = "", user_id: str = "public-user"):
    if not STUDENT_BOT_API_KEY:
        yield {"type": "error", "content": "STUDENT_BOT_API_KEY is not set"}
        return
    payload = {
        "query": query,
        "inputs": {},
        "response_mode": "streaming",
        "conversation_id": conversation_id or "",
        "user": user_id,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STUDENT_BOT_API_KEY}",
    }
    resp = requests.post(DIFY_CHAT_URL, json=payload, headers=headers, verify=False, stream=True, timeout=60)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            line = line[6:]
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = data.get("event")
        if event == "message":
            chunk = data.get("answer", "")
            if chunk:
                yield {"type": "content", "content": chunk}
        elif event == "message_end":
            yield {"type": "done", "conversation_id": data.get("conversation_id", "")}
            return
        elif event == "error":
            yield {"type": "error", "content": data.get("message", "Dify error")}
            return
