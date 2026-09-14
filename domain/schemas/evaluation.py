from typing import List, Optional
from pydantic import BaseModel, Field


class GoldenTestCase(BaseModel):
    """Caso de prueba unitario dentro del Golden Set."""
    id: str = Field(..., description="Identificador único del caso de prueba.")
    pregunta: str = Field(..., description="Pregunta formulada para evaluar el recuperador.")
    documento_esperado: str = Field(..., description="Nombre del archivo documental que contiene la respuesta canónica.")
    categoria: Optional[str] = Field(default=None, description="Categoría temática del caso (ej. 'framework', 'persistence').")
    palabras_clave: List[str] = Field(default_factory=list, description="Palabras clave léxicas relevantes.")


class GoldenSet(BaseModel):
    """Contenedor de todos los casos de prueba del benchmark."""
    version: str = Field(default="1.0.0", description="Versión del benchmark.")
    casos: List[GoldenTestCase] = Field(..., min_length=1, description="Lista de casos de prueba.")


class RetrievalEvaluationResult(BaseModel):
    """Resultado de evaluación para una consulta individual en el benchmark."""
    id: str
    pregunta: str
    documento_esperado: str
    documentos_recuperados: List[str] = Field(
        ...,
        description="Lista de fuentes recuperadas por el sistema en orden de relevancia."
    )
    hit: bool = Field(
        ...,
        description="True si el documento esperado está presente en el Top-k recuperado."
    )
    recall_at_k: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Recall@k para esta consulta (1.0 si fue encontrado, 0.0 si no)."
    )
    precision_at_k: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Precision@k para esta consulta (proporción de documentos relevantes recuperados)."
    )
    latencia_ms: float = Field(
        ...,
        ge=0.0,
        description="Tiempo de respuesta en milisegundos de la recuperación híbrida."
    )


class BenchmarkSummary(BaseModel):
    """Resumen global de la evaluación cuantitativa."""
    total_consultas: int
    k: int
    recall_promedio: float = Field(..., ge=0.0, le=1.0)
    precision_promedio: float = Field(..., ge=0.0, le=1.0)
    tasa_exito_hit: float = Field(..., ge=0.0, le=1.0, description="Porcentaje de consultas con hit=True.")
    latencia_promedio_ms: float
    resultados: List[RetrievalEvaluationResult]
