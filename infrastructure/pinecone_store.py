import time
from typing import Optional, Any, Tuple
from loguru import logger
from pinecone import Pinecone, ServerlessSpec
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.config.settings import get_settings


def get_embedding_model_and_dimension() -> Tuple[Embeddings, int]:
    """
    Instancia el modelo de embeddings adecuado y retorna la tupla (modelo, dimensión).

    Estrategia resiliente y agnóstica:
      - Si OPENAI_API_KEY está configurada y no es un marcador inválido, utiliza
        OpenAI `text-embedding-3-small` (1536 dimensiones).
      - En caso contrario, o si se especifica HuggingFace, utiliza el modelo local
        `sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones, 100% gratuito y local).

    Esto evita el error común #1 de mismatch de dimensiones y asegura ejecución
    inmediata incluso sin suscripción activa a OpenAI.
    """
    settings = get_settings()
    openai_key = settings.openai_api_key

    is_valid_openai_key = bool(openai_key and not openai_key.startswith("invalid_") and len(openai_key) > 20)

    if is_valid_openai_key:
        logger.info("Utilizando proveedor de embeddings OpenAI: text-embedding-3-small (dimensión 1536)")
        model = OpenAIEmbeddings(
            model=settings.openai_embedding_model or "text-embedding-3-small",
            api_key=openai_key,
        )
        return model, 1536
    else:
        logger.info(
            "OPENAI_API_KEY no detectada o inválida. "
            "Utilizando embeddings locales HuggingFace: sentence-transformers/all-MiniLM-L6-v2 (dimensión 384)"
        )
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return model, 384


def get_pinecone_client(api_key: Optional[str] = None) -> Pinecone:
    """Inicializa y retorna el cliente oficial del SDK de Pinecone."""
    settings = get_settings()
    key = api_key or settings.pinecone_api_key
    if not key:
        raise ValueError("No se encontró PINECONE_API_KEY configurada en el entorno o .env.")
    return Pinecone(api_key=key)


def setup_pinecone_index(
    index_name: Optional[str] = None,
    dimension: Optional[int] = None,
    metric: str = "cosine",
    cloud: Optional[str] = None,
    region: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    Verifica si el índice de Pinecone Serverless existe y lo crea si es necesario.
    Si el índice preexistente tiene una dimensión distinta a la requerida por el modelo
    de embeddings, lo recrea para evitar el error crítico de mismatch de dimensiones.

    Args:
        index_name: Nombre del índice en Pinecone.
        dimension: Dimensión del vector (si es None, se deduce del modelo de embeddings).
        metric: Métrica de similitud ('cosine', 'dotproduct', 'euclidean').
        cloud: Proveedor cloud serverless ('aws', 'gcp').
        region: Región serverless (ej. 'us-east-1').
        api_key: Pinecone API Key (opcional, por defecto desde settings).

    Returns:
        Instancia de Index de Pinecone conectada y lista.
    """
    settings = get_settings()
    target_name = index_name or settings.pinecone_index_name
    target_cloud = cloud or settings.pinecone_cloud
    target_region = region or settings.pinecone_region

    if dimension is None:
        _, required_dimension = get_embedding_model_and_dimension()
    else:
        required_dimension = dimension

    pc = get_pinecone_client(api_key=api_key)
    existing_indexes = pc.list_indexes()
    match = next((i for i in existing_indexes if i.name == target_name), None)

    if match is not None:
        # Verificar coincidencia de dimensiones
        if match.dimension != required_dimension:
            logger.warning(
                f"⚠️ Mismatch de dimensiones detectado en índice '{target_name}': "
                f"dimensión actual en Pinecone={match.dimension}, pero el modelo requiere={required_dimension}. "
                f"Recreando índice en modo Serverless para sincronizar dimensiones..."
            )
            pc.delete_index(target_name)
            while target_name in [i.name for i in pc.list_indexes()]:
                time.sleep(1)
            match = None

    if match is None:
        logger.info(
            f"Creando índice Serverless '{target_name}' en Pinecone "
            f"(dimensión={required_dimension}, métrica={metric}, cloud={target_cloud}, region={target_region})..."
        )
        pc.create_index(
            name=target_name,
            dimension=required_dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=target_cloud,
                region=target_region,
            ),
        )
        logger.info(f"Índice '{target_name}' solicitado. Esperando a que esté listo...")
        while not pc.describe_index(target_name).status.get("ready", False):
            time.sleep(1)
        logger.info(f"Índice Serverless '{target_name}' listo y operativo.")
    else:
        logger.info(f"Índice '{target_name}' verificado (dimensión={match.dimension}). Estado listo.")

    return pc.Index(target_name)


def get_pinecone_vector_store(
    index_name: Optional[str] = None,
    embeddings: Optional[Embeddings] = None,
    namespace: Optional[str] = None,
) -> PineconeVectorStore:
    """
    Retorna la integración PineconeVectorStore de LangChain conectada al índice y namespace especificados.
    """
    settings = get_settings()
    target_name = index_name or settings.pinecone_index_name
    target_namespace = namespace or settings.pinecone_namespace

    if embeddings is None:
        emb, dim = get_embedding_model_and_dimension()
    else:
        emb = embeddings
        dim = 1536 if isinstance(emb, OpenAIEmbeddings) else 384

    # Asegurar que el índice exista y coincida en dimensión
    index = setup_pinecone_index(index_name=target_name, dimension=dim)

    return PineconeVectorStore(
        index=index,
        embedding=emb,
        namespace=target_namespace,
    )


def get_namespace_vector_count(
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
) -> int:
    """Retorna la cantidad de vectores indexados en un namespace determinado."""
    settings = get_settings()
    target_name = index_name or settings.pinecone_index_name
    target_namespace = namespace or settings.pinecone_namespace

    pc = get_pinecone_client()
    existing_indexes = [i.name for i in pc.list_indexes()]
    if target_name not in existing_indexes:
        return 0

    index = pc.Index(target_name)
    stats = index.describe_index_stats()
    ns_stats = stats.get("namespaces", {}).get(target_namespace, {})
    return ns_stats.get("vector_count", 0)
