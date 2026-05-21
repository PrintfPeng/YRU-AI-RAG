import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env
load_dotenv()

class LocalLLMProvider:
    @staticmethod
    def get_primary_llm(temperature: float = 0.1):
        """
        เชื่อมต่อ LLM ผ่าน OpenAI-Compatible API ของ Open WebUI
        Temperature ต่ำ (0.1) ตามกฎ Anti-Hallucination
        """
        return ChatOpenAI(
            api_key=os.getenv("OPEN_WEBUI_API_KEY"),
            base_url=os.getenv("OPEN_WEBUI_BASE_URL"),
            model="scb10x/llama3.1-typhoon2-8b-instruct:latest", # ต้องระบุชื่อให้ตรงกับใน Ollama/Open WebUI
            temperature=temperature,
            max_tokens=1024,
            model_kwargs={"top_p": 0.9}
        )

    @staticmethod
    def get_fallback_llm(temperature: float = 0.1):
        """
        โมเดลสำรอง: สำหรับงานง่ายๆ เช่น สรุปผล (Summarization) หรือตกแต่งประโยค
        """
        return ChatOpenAI(
            api_key=os.getenv("OPEN_WEBUI_API_KEY"),
            base_url=os.getenv("OPEN_WEBUI_BASE_URL"),
            model="llama3:latest", 
            temperature=temperature,
            max_tokens=512,
        )

    @staticmethod
    def get_router_llm():
        """
        โมเดลสำหรับ Router (แยกแยะ Intent) แนะนำให้ใช้ตัวที่เล็กและเร็วที่สุด หรือ Qwen อุณหภูมิ 0
        """
        return ChatOpenAI(
            api_key=os.getenv("OPEN_WEBUI_API_KEY"),
            base_url=os.getenv("OPEN_WEBUI_BASE_URL"),
            model="scb10x/llama3.1-typhoon2-8b-instruct:latest", 
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}} # บังคับให้ตอบเป็น JSON
        )
