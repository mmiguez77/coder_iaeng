from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from app.domain.schemas.chat import ChatMessage, ModelParameters, ModelResponse


class BaseLLMClient(ABC):
    """Interfaz abstracta que define el contrato común para todos los clientes de LLM."""

    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa encapsulada en ModelResponse sin arrojar crashes en caso de error de API."""
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Genera tokens uno a uno a través de un generador asíncrono."""
        raise NotImplementedError
        yield  # Indica a Python que el método es un generador asíncrono
