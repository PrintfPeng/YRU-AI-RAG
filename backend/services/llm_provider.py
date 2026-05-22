import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class LocalLLMProvider:
    # Allow model names to be overridden via environment variables
    _PRIMARY_MODEL = os.getenv("LLM_PRIMARY_MODEL", "scb10x/llama3.1-typhoon2-8b-instruct:latest")
    _FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "llama3:latest")

    @staticmethod
    def get_primary_llm(temperature: float = 0.1):
        """
        LLM หลักสำหรับงานทั่วไป (RAG, SQL generation, summarization)
        Temperature ต่ำเพื่อลด Hallucination
        """
        return ChatOpenAI(
            api_key=os.getenv("OPEN_WEBUI_API_KEY"),
            base_url=os.getenv("OPEN_WEBUI_BASE_URL"),
            model=LocalLLMProvider._PRIMARY_MODEL,
            temperature=temperature,
            max_tokens=1024,
        )

    @staticmethod
    def get_fallback_llm(temperature: float = 0.1):
        """LLM สำรองสำหรับงานเบา เช่น สรุปผลสั้น หรือ format ข้อความ"""
        return ChatOpenAI(
            api_key=os.getenv("OPEN_WEBUI_API_KEY"),
            base_url=os.getenv("OPEN_WEBUI_BASE_URL"),
            model=LocalLLMProvider._FALLBACK_MODEL,
            temperature=temperature,
            max_tokens=512,
        )

    @staticmethod
    def get_router_llm():
        """LLM สำหรับ Query Router — ต้องการแค่คำตอบสั้น ('sql' หรือ 'rag')"""
        return ChatOpenAI(
            api_key=os.getenv("OPEN_WEBUI_API_KEY"),
            base_url=os.getenv("OPEN_WEBUI_BASE_URL"),
            model=LocalLLMProvider._PRIMARY_MODEL,
            temperature=0.0,
            max_tokens=10,
        )
