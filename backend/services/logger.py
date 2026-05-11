# backend/services/logger.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import json

# -------------------------------------------------------------------
# System Logging Configuration
# -------------------------------------------------------------------
# กำหนดเส้นทางหลักของระบบ (ชี้ไปที่โฟลเดอร์ backend)
ROOT_DIR = Path(__file__).resolve().parents[1]

# กำหนดโฟลเดอร์และไฟล์สำหรับเก็บ Log 
# เลือกระบบจัดเก็บแบบ JSONL (JSON Lines) แทน JSON ธรรมดา 
# เพื่อประสิทธิภาพสูงสุดในการเขียนข้อมูลต่อท้าย (Streaming Append) โดยไม่ต้องโหลดข้อมูลทั้งก้อนเข้า Memory
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "qa_log.jsonl"

# สร้างโฟลเดอร์ logs อัตโนมัติหากยังไม่มีอยู่ ป้องกันข้อผิดพลาดตอนเริ่มระบบ
LOG_DIR.mkdir(exist_ok=True)


def append_log(entry: Dict[str, Any]) -> None:
    """
    บันทึกข้อมูลเหตุการณ์ (Event) ลงในไฟล์ Log แบบ JSONL (1 บรรทัด = 1 JSON Object)
    
    Args:
        entry (Dict[str, Any]): ข้อมูลที่ต้องการบันทึก ควรประกอบด้วยคีย์หลัก เช่น:
            - query: คำถามจากผู้ใช้
            - answer: คำตอบที่ AI ประมวลผลได้
            - doc_ids: รหัสเอกสารที่ใช้อ้างอิง
            - intent: หมวดหมู่หรือเจตนาของคำถาม
            - mode: โหมดการทำงานของระบบ (เช่น auto, table)
            - sources: แหล่งข้อมูลอ้างอิง
    """
    # คัดลอกข้อมูลเพื่อสร้างตัวแปรอิสระ ป้องกันการดัดแปลงข้อมูลต้นทางที่ถูกส่งเข้ามา
    payload = dict(entry)
    
    # ประทับเวลา (Timestamp) ปัจจุบันในรูปแบบมาตรฐาน ISO 8601 (UTC) อัตโนมัติ 
    # (ใช้ setdefault เพื่อไม่ให้ทับข้อมูลหากมีการส่ง Timestamp ระบุมาแล้ว)
    payload.setdefault("ts", datetime.utcnow().isoformat() + "Z")

    # เปิดไฟล์ในโหมด 'a' (Append) เพื่อเขียนข้อมูลต่อท้ายไฟล์
    # ตั้งค่า ensure_ascii=False เพื่อให้บันทึกอักขระภาษาไทยได้ถูกต้องและอ่านรู้เรื่อง
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    อ่านประวัติการสนทนาย้อนหลัง (Recent Logs) จากไฟล์ เพื่อนำไปสร้าง Memory ให้กับ RAG
    หรือใช้สำหรับการแสดงผลในหน้า History
    
    Args:
        limit (int): จำนวนรายการย้อนหลังสูงสุดที่ต้องการอ่าน (ค่าเริ่มต้น: 50 รายการล่าสุด)
        
    Returns:
        List[Dict[str, Any]]: รายการของ Log ที่แปลงจากข้อความ JSON เป็น Dictionary แล้ว
    """
    # หากยังไม่มีไฟล์ Log (เพิ่งรันระบบครั้งแรก) ให้คืนค่าเป็นลิสต์ว่าง เพื่อป้องกันระบบล่ม (FileNotFoundError)
    if not LOG_FILE.exists():
        return []

    # เปิดไฟล์ในโหมด 'r' (Read) เพื่ออ่านประวัติทั้งหมด
    with LOG_FILE.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # ทำความสะอาดข้อมูล: ตัดช่องว่างส่วนเกินและคัดกรองบรรทัดว่างทิ้ง
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        return []

    # ดึงเฉพาะรายการท้ายสุดตามจำนวนที่กำหนด (Limit) เพื่อความรวดเร็วและประหยัดหน่วยความจำ
    selected = lines[-limit:]
    logs: List[Dict[str, Any]] = []
    
    # แปลงข้อความ JSON Lines แต่ละบรรทัด กลับเป็นโครงสร้าง Dictionary
    for ln in selected:
        try:
            logs.append(json.loads(ln))
        except json.JSONDecodeError:
            # ระบบความปลอดภัย (Graceful Degradation): 
            # ข้ามบรรทัดที่ข้อมูล JSON เสียหาย (Corrupted Data) เพื่อให้ระบบทำงานบรรทัดอื่นต่อไปได้
            continue

    return logs