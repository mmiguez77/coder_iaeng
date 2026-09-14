"""
Script de Inicialización de Infraestructura Pinecone — Módulo 4
================================================================
Verifica la existencia del índice Serverless en Pinecone y lo crea
automáticamente si no existe con la dimensión y especificación requeridas.

Ejecución desde la raíz del repositorio:
    PYTHONPATH=. python app/setup_pinecone.py
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
from app.config.settings import get_settings
from app.infrastructure.pinecone_store import (
    get_pinecone_client,
    setup_pinecone_index,
)

# Configuración de logs
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


def run_setup():
    start_time = time.perf_counter()
    logger.info("=== INICIALIZACIÓN DE INFRAESTRUCTURA PINECONE SERVERLESS ===")

    settings = get_settings()
    if not settings.pinecone_api_key:
        logger.critical("PINECONE_API_KEY no está configurada en .env. Abortando.")
        sys.exit(1)

    logger.info(f"Índice objetivo: {settings.pinecone_index_name}")
    logger.info(f"Cloud/Región: {settings.pinecone_cloud} / {settings.pinecone_region}")
    logger.info("Dimensión vectorial requerida: 1536 (OpenAI text-embedding-3-small)")
    logger.info("Métrica: cosine")

    try:
        index = setup_pinecone_index(
            index_name=settings.pinecone_index_name,
            dimension=1536,
            metric="cosine",
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )

        stats = index.describe_index_stats()
        logger.info(f"Estado e información del índice en Pinecone:\n{stats}")
        logger.info(f"Dimensión configurada: {stats.get('dimension')}")
        logger.info(f"Total de vectores: {stats.get('total_vector_count', 0)}")
        logger.info(f"Namespaces activos: {list(stats.get('namespaces', {}).keys())}")

        elapsed = time.perf_counter() - start_time
        logger.info(f"=== SETUP DE PINECONE COMPLETADO EXITOSAMENTE ({elapsed:.2f}s) ===")
    except Exception as e:
        logger.error(f"Error durante la inicialización de Pinecone: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_setup()
