"""Configuration management for the LangChain Agentic RAG system."""

import os
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    openwebui_api_key: Optional[str] = Field(None, env="OPENWEBUI_API_KEY")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8001, env="PORT")
    debug: bool = Field(False, env="DEBUG")
    
    # Database Configuration
    chroma_db_path: str = Field("./chroma_db", env="CHROMA_DB_PATH")
    
    # RAG Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    max_search_results: int = Field(5, env="MAX_SEARCH_RESULTS")
    
    # LangChain Configuration
    model_name: str = Field("gemini-1.5-pro", env="GEMINI_MODEL_NAME")
    temperature: float = Field(0.7, env="MODEL_TEMPERATURE")
    max_tokens: int = Field(1024, env="MAX_TOKENS")
    
    # Agent Configuration
    max_agent_iterations: int = Field(10, env="MAX_AGENT_ITERATIONS")
    agent_verbose: bool = Field(True, env="AGENT_VERBOSE")
    
    # Session Configuration
    max_conversation_history: int = Field(10, env="MAX_CONVERSATION_HISTORY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()