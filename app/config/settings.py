from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Data storage
    CODERR_DATA_DIR: str = "./coderr_data"

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_MAX_RETRIES: int = 3

    # Embedding
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_BATCH_SIZE: int = 32

    # Retrieval
    MAX_RETRIEVAL_RESULTS: int = 20
    GRAPH_EXPANSION_DEPTH: int = 2
    MAX_CONTEXT_CHARS: int = 8000

    # Logging
    LOG_LEVEL: str = "INFO"

    @property
    def data_path(self) -> Path:
        p = Path(self.CODERR_DATA_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def repo_data_path(self, repo_name: str) -> Path:
        p = self.data_path / repo_name
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
