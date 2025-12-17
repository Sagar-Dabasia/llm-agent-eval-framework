from pydantic import BaseModel
import os

class Settings(BaseModel):
    llm_provider: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    temperature: float = 0.2

def load_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip(),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip(),
        temperature=float(os.getenv("TEMPERATURE", "0.2").strip()),
    )
