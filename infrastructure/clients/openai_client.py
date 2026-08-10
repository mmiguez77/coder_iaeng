from typing import AsyncGenerator, List
from loguru import logger

try:
    from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
except ImportError:
    AsyncOpenAI = None
    APIError = Exception
    RateLimitError = Exception
    APIConnectionError = Exception

from app.domain.entities.base_client import BaseLLMClient
from app.domain.schemas.chat import ChatMessage, ModelResponse, Provider


class OpenAIClient(BaseLLMClient):
    """Implementación del cliente asíncrono para OpenAI."""

    def __init__(self, api_key: str, model_name: str, temperature: float, max_tokens: int):
        super().__init__(model_name, temperature, max_tokens)
        if AsyncOpenAI is None:
            raise ImportError("La librería de openai no está instalada.")
        self._client = AsyncOpenAI(api_key=api_key)

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[dict]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        logger.info(f"OpenAI Client: Generando respuesta para modelo={self.model_name}")
        formatted = self._prepare_messages(messages)
        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=formatted,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model_name,
                content=content,
                success=True,
            )
        except RateLimitError as e:
            logger.error(f"OpenAI Rate limit error: {e}")
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model_name,
                content="",
                error=f"Límite de cuota excedido: {str(e)}",
                success=False,
            )
        except APIConnectionError as e:
            logger.error(f"OpenAI Connection error: {e}")
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model_name,
                content="",
                error=f"Error de conexión a OpenAI: {str(e)}",
                success=False,
            )
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model_name,
                content="",
                error=f"Error de la API de OpenAI: {str(e)}",
                success=False,
            )
        except Exception as e:
            logger.error(f"OpenAI error inesperado: {e}")
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model_name,
                content="",
                error=f"Error inesperado: {str(e)}",
                success=False,
            )

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        logger.info(f"OpenAI Client (Stream): Generando stream para modelo={self.model_name}")
        formatted = self._prepare_messages(messages)
        try:
            stream = await self._client.chat.completions.create(
                model=self.model_name,
                messages=formatted,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except (RateLimitError, APIConnectionError, APIError) as e:
            logger.error(f"OpenAI Stream error: {e}")
            yield f"\n[Error durante el streaming de OpenAI: {str(e)}]"
        except Exception as e:
            logger.error(f"OpenAI Stream error inesperado: {e}")
            yield f"\n[Error inesperado en el streaming de OpenAI: {str(e)}]"
