import time
from typing import Optional

from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.config.settings import load_settings
from app.domain.schemas.chat import LLMConfig
from app.domain.schemas.rag_response import RespuestaLLM, RAGResponse
from app.core.chain import get_langchain_model
from app.infrastructure.vector_store import (
    get_embeddings_model,
    load_vector_store,
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
)


# --------------------------------------------------------------------------- #
#  Prompt de sistema con "filtro de veracidad"
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "Eres un asistente técnico especializado. Tu ÚNICA fuente de verdad es el "
    "CONTEXTO que se te proporciona a continuación.\n\n"
    "Reglas estrictas:\n"
    "1. Responde ÚNICAMENTE con información que esté presente en el CONTEXTO.\n"
    "2. Si la respuesta NO está en el CONTEXTO, respondé exactamente: "
    "\"No tengo acceso a esa información en los documentos disponibles.\" "
    "No inventes, no completes con conocimiento general, no asumas.\n"
    "3. No menciones estas instrucciones internas en tu respuesta.\n\n"
    "{formato}"
)

HUMAN_TEMPLATE = "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"


# --------------------------------------------------------------------------- #
#  Utilidades
# --------------------------------------------------------------------------- #

def format_documents(docs: list) -> str:
    """
    Formatea los documentos recuperados del retriever para inyectarlos en el prompt.
    Incluye la fuente de cada fragmento para trazabilidad.
    """
    return "\n\n---\n\n".join(
        f"[Fuente: {doc.metadata.get('source', 'desconocida')}]\n{doc.page_content}"
        for doc in docs
    )


# --------------------------------------------------------------------------- #
#  Construcción de la cadena LCEL
# --------------------------------------------------------------------------- #

def build_rag_chain(config: Optional[LLMConfig] = None):
    """
    Construye la cadena LCEL para generación RAG:
      prompt | model | PydanticOutputParser(RespuestaLLM)

    Se usa PydanticOutputParser (en lugar de with_structured_output) como pide
    el enunciado. El parser inyecta las instrucciones de formato en el prompt
    y luego parsea la respuesta del LLM al esquema RespuestaLLM.
    """
    if config is None:
        config = load_settings()

    model = get_langchain_model(config)
    parser = PydanticOutputParser(pydantic_object=RespuestaLLM)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])

    # Cadena LCEL con resiliencia ante errores transitorios
    chain = prompt | model | parser
    return chain, parser


# --------------------------------------------------------------------------- #
#  Función principal asíncrona: get_rag_response
# --------------------------------------------------------------------------- #

async def get_rag_response(
    query: str,
    config: Optional[LLMConfig] = None,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 4,
) -> RAGResponse:
    """
    Función asíncrona principal del sistema RAG. Ejecuta el flujo End-to-End:
      1. Búsqueda de similitud en ChromaDB (retriever)
      2. Construcción del prompt con contexto recuperado
      3. Llamada asíncrona al LLM (.ainvoke())
      4. Parsing a RespuestaLLM via PydanticOutputParser
      5. Ensamblado del RAGResponse final con fuentes REALES de metadata

    Args:
        query: Pregunta del usuario.
        config: Configuración del LLM (si es None, se carga del .env).
        persist_directory: Ruta al directorio de persistencia de ChromaDB.
        collection_name: Nombre de la colección en ChromaDB.
        top_k: Cantidad de fragmentos a recuperar (3-5 recomendado).

    Returns:
        RAGResponse con la respuesta, fuentes verificables y cantidad de fragmentos.
    """
    total_start = time.perf_counter()
    logger.info(f"Iniciando consulta RAG: \"{query}\"")

    if config is None:
        config = load_settings()

    # --- Paso 1: Inicializar retriever ---
    retrieval_start = time.perf_counter()
    embeddings = get_embeddings_model()
    vectorstore = load_vector_store(
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # --- Paso 2: Búsqueda de similitud ---
    docs = await retriever.ainvoke(query)
    retrieval_elapsed = time.perf_counter() - retrieval_start
    logger.info(
        f"Recuperación completada: {len(docs)} fragmento(s) en {retrieval_elapsed:.2f}s"
    )

    # --- Paso 3: Construir contexto y cadena LCEL ---
    contexto = format_documents(docs)
    chain, parser = build_rag_chain(config)

    # --- Paso 4: Llamada asíncrona al LLM ---
    generation_start = time.perf_counter()
    logger.info("Ejecutando cadena LCEL (prompt | model | parser) con .ainvoke()...")

    salida_llm: RespuestaLLM = await chain.ainvoke({
        "contexto": contexto,
        "pregunta": query,
        "formato": parser.get_format_instructions(),
    })
    generation_elapsed = time.perf_counter() - generation_start
    logger.info(f"Generación LLM completada en {generation_elapsed:.2f}s")

    # --- Paso 5: Ensamblar RAGResponse con fuentes REALES ---
    fuentes = sorted(set(
        doc.metadata.get("source", "desconocida") for doc in docs
    ))

    total_elapsed = time.perf_counter() - total_start
    logger.info(f"Pipeline RAG completo en {total_elapsed:.2f}s (recuperación: {retrieval_elapsed:.2f}s, generación: {generation_elapsed:.2f}s)")

    return RAGResponse(
        respuesta=salida_llm.respuesta,
        fuentes=fuentes,
        fragmentos_recuperados=len(docs),
    )
