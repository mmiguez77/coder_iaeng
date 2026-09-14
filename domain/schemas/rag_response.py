from pydantic import BaseModel, Field


class RespuestaLLM(BaseModel):
    """
    Esquema de salida que el PydanticOutputParser parsea directamente del LLM.
    Se mantiene simple para minimizar errores de parsing: solo contiene el texto
    de la respuesta. Las fuentes y metadatos se arman por código, no por el LLM.
    """
    respuesta: str = Field(
        ...,
        min_length=1,
        description=(
            "Respuesta a la pregunta del usuario, basada EXCLUSIVAMENTE en el CONTEXTO. "
            "Si la información no está en el contexto, decir explícitamente que no se "
            "cuenta con esa información."
        )
    )


class RAGResponse(BaseModel):
    """
    Objeto final que devuelve get_rag_response(). Combina la salida del LLM
    con metadata verificable extraída directamente de ChromaDB (no generada por el modelo).
    """
    respuesta: str = Field(
        ...,
        description="Texto de la respuesta generada por el LLM."
    )
    fuentes: list[str] = Field(
        ...,
        description="Lista de archivos de origen de los fragmentos usados como contexto."
    )
    fragmentos_recuperados: int = Field(
        ...,
        ge=0,
        description="Cantidad de fragmentos recuperados del vector store para esta consulta."
    )
