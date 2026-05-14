# ใช้ Python 3.11 เป็นฐานในการรัน
FROM python:3.11-slim

# ตั้งค่า Directory หลักใน Container
WORKDIR /app

# ติดตั้งเครื่องมือพื้นฐานของระบบที่จำเป็น (build-essential สำหรับ ChromaDB)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# คัดลอกไฟล์ทั้งหมดจากโปรเจกต์เข้าไปใน Container
COPY . /app/

# ===== หมายเหตุ: ไม่ติดตั้ง sentence-transformers / torch =====
# ระบบนี้ใช้ Embedding Model ผ่าน Ollama API บน Server มหาลัย (bge-m3:latest)
# ผ่าน langchain-openai (OpenAIEmbeddings compatible endpoint)
# จึงไม่จำเป็นต้องโหลด PyTorch (532MB) ลงในเครื่อง

# ติดตั้ง Library ที่จำเป็นสำหรับ Hybrid RAG (Slim Version - ไม่มี PyTorch/Torch)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-dotenv \
    mysql-connector-python \
    langchain-core \
    langchain-community \
    langchain-ollama \
    langchain-openai \
    chromadb \
    pythainlp \
    pydantic \
    requests \
    python-multipart

# เปิด Port 8005 เพื่อให้ภายนอกเข้าถึง FastAPI ได้
EXPOSE 8005

# รัน Backend Server ทันทีที่ Container เริ่มทำงาน
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8005"]
