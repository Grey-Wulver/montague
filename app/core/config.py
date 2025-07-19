# app/core/config.py
from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # Database
    database_url: str = "sqlite:///./data/netops.db"

    # OpenAI
    openai_api_key: Optional[str] = None

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Network Automation
    default_timeout: int = 30
    max_concurrent_devices: int = 10

    # LLM Configuration
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "codellama:13b"
    llm_enabled: bool = True
    llm_timeout: int = 30
    llm_max_retries: int = 2
    normalization_cache_size: int = 1000

    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    return settings
