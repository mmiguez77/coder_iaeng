from typing import Optional
from loguru import logger

from app.domain.entities.base_client import BaseLLMClient
from app.domain.schemas.chat import LLMConfig, Provider
from app.infrastructure.clients.openai_client import OpenAIClient
from app.infrastructure.clients.anthropic_client import AnthropicClient
from app.infrastructure.clients.gemini_client import GeminiClient


class LLMFactory:
    """Clase fábrica para instanciar clientes de LLM de forma limpia según su configuración."""

    @staticmethod
    def create_client(
        provider: Provider, config: LLMConfig, model_name: Optional[str] = None
    ) -> BaseLLMClient:
        """
        Instancia un cliente LLM que cumple con el contrato BaseLLMClient.

        Args:
            provider: El proveedor a instanciar (openai, anthropic, gemini).
            config: La configuración global con las credenciales validadas.
            model_name: Opcional, nombre de modelo específico.

        Returns:
            Una instancia concreta de BaseLLMClient.
        """
        temp = config.temperature
        max_tok = config.max_tokens

        if provider == Provider.OPENAI:
            if not config.openai_api_key:
                raise ValueError("Falta openai_api_key en la configuración.")
            model = model_name or "gpt-4o-mini"
            logger.debug(f"LLMFactory: Instanciando OpenAIClient con modelo={model}")
            return OpenAIClient(
                api_key=config.openai_api_key.get_secret_value(),
                model_name=model,
                temperature=temp,
                max_tokens=max_tok,
            )

        elif provider == Provider.ANTHROPIC:
            if not config.anthropic_api_key:
                raise ValueError("Falta anthropic_api_key en la configuración.")
            model = model_name or "claude-3-5-sonnet-20241022"
            logger.debug(f"LLMFactory: Instanciando AnthropicClient con modelo={model}")
            return AnthropicClient(
                api_key=config.anthropic_api_key.get_secret_value(),
                model_name=model,
                temperature=temp,
                max_tokens=max_tok,
            )

        elif provider == Provider.GEMINI:
            if not config.gemini_api_key:
                raise ValueError("Falta gemini_api_key en la configuración.")
            # Para Gemini Studio, gemini-2.5-flash es el modelo por defecto y estable actual.
            model = model_name or "gemini-2.5-flash"
            logger.debug(f"LLMFactory: Instanciando GeminiClient con modelo={model}")
            return GeminiClient(
                api_key=config.gemini_api_key.get_secret_value(),
                model_name=model,
                temperature=temp,
                max_tokens=max_tok,
            )

        else:
            raise ValueError(f"Proveedor '{provider}' no está soportado por la fábrica.")
