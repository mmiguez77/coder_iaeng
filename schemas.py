"""
Módulo schemas.py - Re-exporta los esquemas de Pydantic para el Pipeline de Extracción de Entidades Técnicas.
"""
from app.domain.schemas.technical_entity import (
    CriticidadEnum,
    TechnicalEntityExtraction,
)

__all__ = ["CriticidadEnum", "TechnicalEntityExtraction"]
