import os
from pathlib import Path
from typing import Optional
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.domain.schemas.chat import LLMConfig, Provider


class AppSettings(BaseSettings):
    """Clase de configuración que carga las variables de entorno."""
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parents[1] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    default_provider: Optional[str] = None
    default_model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 1024

    # Configuración de Pinecone (Módulo 4)
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "coderhouse-rag-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "dev-technical-docs"
    openai_embedding_model: str = "text-embedding-3-small"

    def get_llm_config(self) -> LLMConfig:
        """Retorna la configuración validada de LLMConfig de Pydantic."""
        provider = None
        if self.default_provider:
            try:
                provider = Provider(self.default_provider.lower())
            except ValueError:
                provider = None

        return LLMConfig(
            openai_api_key=SecretStr(self.openai_api_key) if self.openai_api_key else None,
            anthropic_api_key=SecretStr(self.anthropic_api_key) if self.anthropic_api_key else None,
            gemini_api_key=SecretStr(self.gemini_api_key) if self.gemini_api_key else None,
            default_provider=provider,
            default_model=self.default_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


def get_settings() -> AppSettings:
    """Retorna la instancia completa de configuración AppSettings."""
    return AppSettings()


def load_settings() -> LLMConfig:
    """Carga y retorna la configuración de la aplicación para LLMs (compatibilidad Módulo 2 y 3)."""
    settings = AppSettings()
    return settings.get_llm_config()
