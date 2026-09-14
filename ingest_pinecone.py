"""
Pipeline de Ingesta Inteligente a Pinecone Serverless — Módulo 4
==================================================================
Lee documentos técnicos (.md) de app/data/, aplica chunking basado en tokens,
enriquece cada fragmento con metadatos avanzados y los inserta con embeddings
en el namespace de Pinecone.

Soporta tanto OpenAIEmbeddings (text-embedding-3-small, 1536 dims) como
HuggingFaceEmbeddings locales (all-MiniLM-L6-v2, 384 dims) de forma transparente
según las credenciales configuradas en .env.

Ejecución desde la raíz del repositorio:
    PYTHONPATH=. python app/ingest_pinecone.py [--force]
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

# Asegurar que los módulos de 'app' sean importables independientemente del CWD
app_dir = Path(__file__).resolve().parent
parent_dir = app_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from loguru import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from app.config.settings import get_settings
from app.infrastructure.pinecone_store import (
    get_embedding_model_and_dimension,
    setup_pinecone_index,
    get_namespace_vector_count,
)

# --------------------------------------------------------------------------- #
#  Configuración de logs
# --------------------------------------------------------------------------- #
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "execution.log"

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:DD/MM/YY - HH.mm.ss}</green> | <level>{level: <8}</level> | {function}:{line} - {message}",
    level="INFO",
)
logger.add(
    str(log_file_path),
    format="{time:DD/MM/YY - HH.mm.ss} | {level: <8} | {function}:{line} - {message}",
    level="INFO",
    rotation="10 MB",
)

# --------------------------------------------------------------------------- #
#  Configuración de parámetros de ingesta
# --------------------------------------------------------------------------- #
DATA_DIR = str(Path(__file__).parent / "data")
CHUNK_SIZE = 500       # Tamaño objetivo en tokens (~500-800 recomendado)
CHUNK_OVERLAP = 70     # Solapamiento para mantener coherencia semántica en los bordes
BATCH_SIZE = 50        # Tamaño de lote para upsert en Pinecone

CATEGORY_MAPPING: Dict[str, str] = {
    "langchain_intro.md": "framework",
    "pydantic_guide.md": "data-validation",
    "chromadb_guide.md": "persistence",
    "rag_concepts.md": "architecture",
    "pinecone_serverless_guide.md": "cloud-infrastructure",
}


def load_raw_documents() -> List[Document]:
    """Carga los documentos markdown desde la carpeta data/."""
    logger.info(f"Cargando documentos desde '{DATA_DIR}'...")
    loader = DirectoryLoader(
        DATA_DIR,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()
    logger.info(f"Total de documentos fuente cargados: {len(raw_docs)}")
    return raw_docs


def split_and_enrich_documents(raw_docs: List[Document]) -> List[Document]:
    """
    Divide los documentos en chunks optimizados por tokens y enriquece cada chunk
    con metadatos avanzados: fuente normalizada, categoría, índice de fragmento,
    recuento de caracteres y marca temporal de ingesta.
    """
    logger.info(
        f"Aplicando RecursiveCharacterTextSplitter (tiktoken): "
        f"chunk_size={CHUNK_SIZE} tokens, chunk_overlap={CHUNK_OVERLAP} tokens"
    )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    enriched_chunks: List[Document] = []
    ingested_at = datetime.now(timezone.utc).isoformat()

    for doc in raw_docs:
        doc_chunks = splitter.split_documents([doc])
        raw_source = Path(doc.metadata.get("source", "unknown")).name
        category = CATEGORY_MAPPING.get(raw_source, "general-technical")

        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata = {
                "source": raw_source,
                "category": category,
                "chunk_index": idx,
                "total_chunks": len(doc_chunks),
                "char_count": len(chunk.page_content),
                "ingested_at": ingested_at,
                "doc_id": f"{raw_source}#chunk_{idx}",
            }
            enriched_chunks.append(chunk)

    logger.info(f"Total de fragmentos generados y enriquecidos: {len(enriched_chunks)}")
    return enriched_chunks


def run_ingestion(force: bool = False):
    """
    Ejecuta el pipeline completo de ingesta a Pinecone Serverless:
      1. Obtención de embeddings y sincronización de dimensión con el índice
      2. Carga de dataset técnico
      3. Fragmentación y enriquecimiento de metadatos
      4. Vectorización e inserción en batch en el namespace configurado
    """
    start_total = time.perf_counter()
    logger.info("=== INICIANDO PIPELINE DE INGESTA A PINECONE SERVERLESS (MÓDULO 4) ===")

    settings = get_settings()
    index_name = settings.pinecone_index_name
    namespace = settings.pinecone_namespace

    # Paso 1: Resolver embeddings y dimensión correspondiente
    embeddings_model, required_dimension = get_embedding_model_and_dimension()

    # Paso 2: Asegurar que el índice exista y coincida en dimensión
    index = setup_pinecone_index(index_name=index_name, dimension=required_dimension)

    # Verificar si el namespace ya contiene vectores
    current_count = get_namespace_vector_count(index_name=index_name, namespace=namespace)
    logger.info(f"Vectores actuales en namespace '{namespace}': {current_count}")

    if current_count > 0 and not force:
        logger.info(
            f"♻️ El namespace '{namespace}' ya cuenta con {current_count} vectores. "
            "Omitiendo re-ingesta para optimizar costos y evitar duplicados. "
            "(Usa el argumento '--force' para sobreescribir)."
        )
        return

    if force and current_count > 0:
        logger.warning(f"Modo '--force' activado: purgando vectores existentes en namespace '{namespace}'...")
        try:
            index.delete(delete_all=True, namespace=namespace)
            time.sleep(2)
            logger.info("Namespace purgado exitosamente.")
        except Exception as e:
            logger.warning(f"No se pudieron eliminar vectores previos: {e}")

    # Paso 3: Cargar documentos
    raw_docs = load_raw_documents()
    if not raw_docs:
        logger.error("No se encontraron documentos en data/. Abortando.")
        sys.exit(1)

    # Paso 4: Chunking y metadatos avanzados
    chunk_start = time.perf_counter()
    chunks = split_and_enrich_documents(raw_docs)
    chunk_elapsed = time.perf_counter() - chunk_start

    # Paso 5: Generar embeddings e insertar en Pinecone
    logger.info(
        f"Insertando {len(chunks)} fragmentos en Pinecone (Índice: '{index_name}', "
        f"Namespace: '{namespace}') en lotes de {BATCH_SIZE}..."
    )

    upsert_start = time.perf_counter()
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings_model,
        namespace=namespace,
    )

    # Inserción por lotes
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        logger.info(f"  Upserting lote {i // BATCH_SIZE + 1} ({len(batch)} fragmentos)...")
        vector_store.add_documents(documents=batch)

    upsert_elapsed = time.perf_counter() - upsert_start

    # Verificación final
    time.sleep(2)
    final_count = get_namespace_vector_count(index_name=index_name, namespace=namespace)
    total_elapsed = time.perf_counter() - start_total

    logger.info(
        f"=== INGESTA FINALIZADA EXITOSAMENTE ===\n"
        f"  - Total fragmentos insertados: {len(chunks)}\n"
        f"  - Vectores reportados en Pinecone namespace '{namespace}': {final_count}\n"
        f"  - Tiempo de chunking: {chunk_elapsed:.2f}s\n"
        f"  - Tiempo de embeddings y upsert: {upsert_elapsed:.2f}s\n"
        f"  - Tiempo total: {total_elapsed:.2f}s"
    )


if __name__ == "__main__":
    force_run = "--force" in sys.argv
    run_ingestion(force=force_run)
