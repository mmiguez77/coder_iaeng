from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, SecretStr


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class RoleEnum(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Representación de un mensaje en el historial de chat con validación de roles de dominio."""
    role: RoleEnum = Field(
        ...,
        description="El rol de quien emite el mensaje (system, user o assistant)."
    )
    content: str = Field(
        ...,
        min_length=1,
        description="El texto del mensaje."
    )


class ModelParameters(BaseModel):
    """Parámetros de ejecución configurables para los LLMs."""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    """Esquema de configuración global y validación de llaves."""
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    gemini_api_key: Optional[SecretStr] = None
    default_provider: Optional[Provider] = None  # Si es None, indica que se consulta a los tres
    default_model: str = "gpt-4o"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)


class ModelResponse(BaseModel):
    """Respuesta unificada de los clientes que encapsula el resultado o el error."""
    provider: Provider
    model: str
    content: str
    error: Optional[str] = None
    success: bool = True
