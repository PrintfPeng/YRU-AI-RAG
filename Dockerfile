# ใช้ Python 3.11 เป็นฐานในการรัน
FROM python:3.11-slim

# ตั้งค่า Directory หลักใน Container
WORKDIR /app

# ติดตั้งเครื่องมือพื้นฐานของระบบที่จำเป็นสำหรับการลงบาง Library (เช่น ChromaDB)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# คัดลอกไฟล์ทั้งหมดจากโปรเจกต์เข้าไปใน Container
COPY . /app/

# ติดตั้ง Library ที่จำเป็นทั้งหมดสำหรับ Hybrid RAG
RUN pip install --no-cache-dir fastapi uvicorn python-dotenv mysql-connector-python langchain-core langchain-community langchain-huggingface chromadb pythainlp sentence-transformers pydantic langchain-ollama langchain-openai requests python-multipart

# เปิด Port 8000 เพื่อให้ภายนอกเข้าถึง FastAPI ได้
EXPOSE 8000

# รัน Backend Server ทันทีที่ Container เริ่มทำงาน
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
