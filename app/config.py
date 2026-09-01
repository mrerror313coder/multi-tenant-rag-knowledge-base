"""Application configuration and environment settings."""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "Multi-Tenant RAG Knowledge Base"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    SEED_DEMO_DATA: bool = False  # Set to True only if demo organizations should be created automatically

    # Database
    DATABASE_URL: str = "sqlite:///./rag_metadata.db"

    # Vector Database
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # LLM & Embedding Providers
    PRIMARY_LLM_PROVIDER: str = "groq"  # "gemini", "groq", "cohere", "huggingface", "openai", "mock"
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None

    # Embedding Engine: "sentence_transformers", "cohere", "huggingface"
    EMBEDDING_PROVIDER: str = "sentence_transformers"
    COHERE_EMBEDDING_MODEL: str = "embed-multilingual-v3.0"
    HUGGINGFACE_EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # Chunking & Retrieval Hyperparameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3
    SCORE_THRESHOLD: float = 0.05

    # Prompts Directory
    PROMPTS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "services", "prompts")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
