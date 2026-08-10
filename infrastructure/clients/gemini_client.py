from typing import AsyncGenerator, List, Optional
from loguru import logger

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from app.domain.entities.base_client import BaseLLMClient
from app.domain.schemas.chat import ChatMessage, ModelResponse, Provider


class GeminiClient(BaseLLMClient):
    """Implementación del cliente asíncrono para Gemini usando google-genai."""

    def __init__(self, api_key: str, model_name: str, temperature: float, max_tokens: int):
        super().__init__(model_name, temperature, max_tokens)
        if genai is None:
            raise ImportError("La librería de google-genai no está instalada.")
        self._client = genai.Client(api_key=api_key)

    def _convertir_mensajes(self, messages: List[ChatMessage]):
        """Gemini separa la system_instruction de los contents y requiere el rol 'model' para el asistente."""
        contents = []
        system_instruction = None
        for m in messages:
            if m.role.value == "system":
                system_instruction = m.content
            else:
                rol_gemini = "model" if m.role.value == "assistant" else "user"
                contents.append(types.Content(role=rol_gemini, parts=[types.Part(text=m.content)]))
        return contents, system_instruction

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        logger.info(f"Gemini Client: Generando respuesta para modelo={self.model_name}")
        contents, system_instruction = self._convertir_mensajes(messages)
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    system_instruction=system_instruction,
                ),
            )
            return ModelResponse(
                provider=Provider.GEMINI,
                model=self.model_name,
                content=response.text or "",
                success=True,
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return ModelResponse(
                provider=Provider.GEMINI,
                model=self.model_name,
                content="",
                error=f"Error de la API de Gemini: {str(e)}",
                success=False,
            )

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        logger.info(f"Gemini Client (Stream): Generando stream para modelo={self.model_name}")
        contents, system_instruction = self._convertir_mensajes(messages)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    system_instruction=system_instruction,
                ),
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini Stream error: {e}")
            yield f"\n[Error durante el streaming de Gemini: {str(e)}]"
