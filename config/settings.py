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


def load_settings() -> LLMConfig:
    """Carga y retorna la configuración de la aplicación."""
    settings = AppSettings()
    return settings.get_llm_config()
