from typing import AsyncGenerator, List, Optional
from loguru import logger

try:
    from anthropic import (
        AsyncAnthropic,
        APIError as AnthropicAPIError,
        RateLimitError as AnthropicRateLimitError,
        APIConnectionError as AnthropicConnectionError,
    )
except ImportError:
    AsyncAnthropic = None
    AnthropicAPIError = Exception
    AnthropicRateLimitError = Exception
    AnthropicConnectionError = Exception

from app.domain.entities.base_client import BaseLLMClient
from app.domain.schemas.chat import ChatMessage, ModelResponse, Provider


class AnthropicClient(BaseLLMClient):
    """Implementación del cliente asíncrono para Anthropic (Claude)."""

    def __init__(self, api_key: str, model_name: str, temperature: float, max_tokens: int):
        super().__init__(model_name, temperature, max_tokens)
        if AsyncAnthropic is None:
            raise ImportError("La librería de anthropic no está instalada.")
        self._client = AsyncAnthropic(api_key=api_key)

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[dict]:
        # Anthropic no permite el rol 'system' dentro de messages, debe ser alternado.
        # Aquí filtramos los system messages
        return [{"role": "user" if m.role.value == "user" else "assistant", "content": m.content} 
                for m in messages if m.role.value != "system"]

    def _get_system_instruction(self, messages: List[ChatMessage]) -> Optional[str]:
        system_msgs = [m.content for m in messages if m.role.value == "system"]
        return "\n".join(system_msgs) if system_msgs else None

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        logger.info(f"Anthropic Client: Generando respuesta para modelo={self.model_name}")
        formatted = self._prepare_messages(messages)
        system_instruction = self._get_system_instruction(messages)

        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": formatted,
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        try:
            response = await self._client.messages.create(**kwargs)
            content = "".join([block.text for block in response.content if hasattr(block, "text")])
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model_name,
                content=content,
                success=True,
            )
        except AnthropicRateLimitError as e:
            logger.error(f"Anthropic Rate limit error: {e}")
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model_name,
                content="",
                error=f"Límite de cuota excedido: {str(e)}",
                success=False,
            )
        except AnthropicConnectionError as e:
            logger.error(f"Anthropic Connection error: {e}")
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model_name,
                content="",
                error=f"Error de conexión a Anthropic: {str(e)}",
                success=False,
            )
        except AnthropicAPIError as e:
            logger.error(f"Anthropic API error: {e}")
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model_name,
                content="",
                error=f"Error de la API de Anthropic: {str(e)}",
                success=False,
            )
        except Exception as e:
            logger.error(f"Anthropic error inesperado: {e}")
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model_name,
                content="",
                error=f"Error inesperado: {str(e)}",
                success=False,
            )

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        logger.info(f"Anthropic Client (Stream): Generando stream para modelo={self.model_name}")
        formatted = self._prepare_messages(messages)
        system_instruction = self._get_system_instruction(messages)

        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": formatted,
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicAPIError) as e:
            logger.error(f"Anthropic Stream error: {e}")
            yield f"\n[Error durante el streaming de Anthropic: {str(e)}]"
        except Exception as e:
            logger.error(f"Anthropic Stream error inesperado: {e}")
            yield f"\n[Error inesperado en el streaming de Anthropic: {str(e)}]"
        
