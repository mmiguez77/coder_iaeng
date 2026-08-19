from enum import Enum
from pydantic import BaseModel, Field


class CriticidadEnum(str, Enum):
    """Nivel de criticidad para las entidades técnicas extraídas."""
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class TechnicalEntityExtraction(BaseModel):
    """
    Modelo de Pydantic que define la estructura validada de salida
    del Pipeline de Extracción de Entidades Técnicas.
    """
    tecnologias: list[str] = Field(
        ...,
        min_length=1,
        description="Lista de tecnologías, herramientas o lenguajes identificados en el texto (no puede estar vacía)."
    )
    nivel_de_criticidad: CriticidadEnum = Field(
        ...,
        description="Nivel de criticidad evaluado para el caso analizado (baja, media, alta)."
    )
    resumen_tecnico: str = Field(
        ...,
        min_length=5,
        description="Resumen técnico conciso del problema, arquitectura o estado descrito."
    )
