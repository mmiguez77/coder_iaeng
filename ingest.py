"""
Script de Ingesta — Módulo 3: Sistema RAG Local
================================================
Lee los documentos de app/data/, aplica chunking basado en tokens y los persiste
en una colección local de ChromaDB.

Ejecución desde la raíz del repositorio:
    PYTHONPATH=. python app/ingest.py
"""
import sys
import time
from pathlib import Path

# Asegurar que los módulos de 'app' sean importables independientemente del CWD
app_dir = Path(__file__).resolve().parent
parent_dir = app_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from loguru import logger
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.infrastructure.vector_store import (
    get_embeddings_model,
    collection_is_populated,
    index_documents,
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
)

# --------------------------------------------------------------------------- #
#  Configuración de logs (misma convención que main.py del Módulo 2)
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
#  Configuración de ingesta
# --------------------------------------------------------------------------- #
DATA_DIR = str(Path(__file__).parent / "data")
CHUNK_SIZE = 500       # Tamaño en tokens (via tiktoken)
CHUNK_OVERLAP = 70     # Overlap en tokens


def run_ingestion():
    """
    Flujo de ingesta End-to-End:
      1. Carga documentos .md de app/data/
      2. Aplica RecursiveCharacterTextSplitter basado en tokens
      3. Verifica si la colección ya existe (evita re-indexación)
      4. Indexa en ChromaDB con persistencia local
    """
    total_start = time.perf_counter()
    logger.info("=== INICIANDO INGESTA DE DOCUMENTOS (MÓDULO 3 — RAG) ===")

    # --- Paso 1: Cargar documentos ---
    logger.info(f"Cargando documentos .md desde '{DATA_DIR}'...")
    loader = DirectoryLoader(
        DATA_DIR,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()
    logger.info(f"Documentos cargados: {len(raw_docs)}")

    if not raw_docs:
        logger.error("No se encontraron documentos .md en la carpeta data/. Abortando.")
        sys.exit(1)

    for doc in raw_docs:
        logger.info(f"  → {doc.metadata.get('source', 'desconocido')} ({len(doc.page_content)} caracteres)")

    # --- Paso 2: Chunking basado en tokens ---
    chunking_start = time.perf_counter()
    logger.info(
        f"Aplicando RecursiveCharacterTextSplitter (tiktoken): "
        f"chunk_size={CHUNK_SIZE} tokens, chunk_overlap={CHUNK_OVERLAP} tokens"
    )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(raw_docs)
    chunking_elapsed = time.perf_counter() - chunking_start
    logger.info(f"Fragmentos generados: {len(chunks)} (chunking en {chunking_elapsed:.2f}s)")

    # --- Paso 3: Verificar si ya fue indexado ---
    embeddings = get_embeddings_model()

    if collection_is_populated(
        persist_directory=DEFAULT_PERSIST_DIR,
        collection_name=DEFAULT_COLLECTION_NAME,
        embeddings=embeddings,
    ):
        logger.info("♻️  Índice existente detectado — omitiendo re-indexación.")
        logger.info("    Para forzar re-indexación, elimine la carpeta 'vectorstore/'.")
        total_elapsed = time.perf_counter() - total_start
        logger.info(f"=== INGESTA FINALIZADA (ya indexado) — Tiempo total: {total_elapsed:.2f}s ===")
        return

    # --- Paso 4: Indexar en ChromaDB ---
    indexing_start = time.perf_counter()
    index_documents(
        chunks=chunks,
        embeddings=embeddings,
        persist_directory=DEFAULT_PERSIST_DIR,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    indexing_elapsed = time.perf_counter() - indexing_start
    logger.info(f"Indexación completada en {indexing_elapsed:.2f}s")

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        f"=== INGESTA FINALIZADA — Tiempo total: {total_elapsed:.2f}s "
        f"(chunking: {chunking_elapsed:.2f}s, indexación: {indexing_elapsed:.2f}s) ==="
    )


if __name__ == "__main__":
    run_ingestion()
