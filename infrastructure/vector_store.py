import os
from loguru import logger
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Nombre de la colección por defecto para el sistema RAG
DEFAULT_COLLECTION_NAME = "coderhouse_rag"

# Directorio de persistencia relativo a la carpeta app/
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vectorstore")


def get_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Instancia el modelo de embeddings de HuggingFace (ejecución local, gratuito).

    Se centraliza en una sola función para garantizar que SIEMPRE se use el mismo
    modelo tanto para indexar como para consultar, evitando el error #1 del enunciado
    (embeddings no coincidentes).

    Args:
        model_name: Nombre del modelo sentence-transformers a utilizar.

    Returns:
        Instancia de HuggingFaceEmbeddings lista para usar.
    """
    logger.info(f"Inicializando modelo de embeddings: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


def collection_is_populated(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings: HuggingFaceEmbeddings = None,
) -> bool:
    """
    Verifica si la colección ChromaDB ya fue indexada con documentos.
    Evita re-indexación innecesaria optimizando tiempo y costo.

    Returns:
        True si la colección existe y tiene al menos un documento.
    """
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        return False

    try:
        if embeddings is None:
            embeddings = get_embeddings_model()

        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        count = vectorstore._collection.count()
        logger.debug(f"Colección '{collection_name}' contiene {count} documento(s).")
        return count > 0
    except Exception as e:
        logger.warning(f"Error al verificar colección existente: {e}")
        return False


def index_documents(
    chunks: list,
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """
    Indexa una lista de documentos fragmentados en ChromaDB con persistencia local.

    Args:
        chunks: Lista de objetos Document (resultado del text splitter).
        embeddings: Modelo de embeddings (debe ser el mismo usado para consultas).
        persist_directory: Carpeta donde se persiste la base vectorial.
        collection_name: Nombre de la colección en ChromaDB.

    Returns:
        Instancia de Chroma con los documentos indexados.
    """
    logger.info(f"Indexando {len(chunks)} fragmentos en colección '{collection_name}'...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    count = vectorstore._collection.count()
    logger.info(f"Indexación completada. Documentos en colección: {count}")
    return vectorstore


def load_vector_store(
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """
    Carga un vector store ChromaDB existente desde disco.

    Args:
        embeddings: Modelo de embeddings (mismo que se usó para indexar).
        persist_directory: Carpeta donde está persistida la base vectorial.
        collection_name: Nombre de la colección.

    Returns:
        Instancia de Chroma conectada a la colección existente.
    """
    logger.info(f"Cargando vector store desde '{persist_directory}', colección '{collection_name}'...")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    count = vectorstore._collection.count()
    logger.info(f"Vector store cargado. Documentos en colección: {count}")
    return vectorstore
