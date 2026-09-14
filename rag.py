"""
Script de Prueba RAG — Módulo 3: Sistema RAG Local
====================================================
Ejecuta dos casos de prueba sobre el sistema RAG asíncrono:
  - Prueba 1 (positiva): Pregunta cuya respuesta está en los documentos indexados.
  - Prueba 2 (trampa): Pregunta cuya respuesta NO está, verificando que el modelo no alucine.

Ejecución desde la raíz del repositorio:
    PYTHONPATH=. python app/rag.py
"""
import asyncio
import json
import sys
from pathlib import Path

# Asegurar que los módulos de 'app' sean importables independientemente del CWD
app_dir = Path(__file__).resolve().parent
parent_dir = app_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from loguru import logger

from app.config.settings import load_settings
from app.core.rag_chain import get_rag_response

# --------------------------------------------------------------------------- #
#  Configuración de logs (misma convención que main.py y ingest.py)
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
#  Casos de prueba
# --------------------------------------------------------------------------- #

async def run_rag_tests():
    """
    Ejecuta dos pruebas sobre el sistema RAG:
      1. Pregunta con respuesta en el contexto → respuesta fundamentada
      2. Pregunta trampa sin respuesta en el contexto → "No tengo acceso..."
    """
    logger.info("=== INICIANDO PRUEBAS DEL SISTEMA RAG (MÓDULO 3) ===")

    try:
        config = load_settings()
        logger.info("Configuración cargada correctamente desde las variables de entorno.")
    except Exception as e:
        logger.critical(f"Error al cargar las configuraciones: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------- #
    #  PRUEBA 1: Pregunta con respuesta en el contexto
    # ------------------------------------------------------------------- #
    query_1 = "¿Qué es LCEL en LangChain y cómo se compone una cadena?"

    logger.info("\n--- PRUEBA 1: Pregunta con respuesta en el contexto ---")
    logger.info(f"Consulta: \"{query_1}\"")

    try:
        result_1 = await get_rag_response(query_1, config=config)
        json_output_1 = json.dumps(result_1.model_dump(), indent=2, ensure_ascii=False)
        logger.info(f"RAGResponse validado (JSON):\n{json_output_1}")
        print("\n--- SALIDA JSON (PRUEBA 1 — Respuesta esperada en contexto) ---")
        print(json_output_1)
    except Exception as e:
        logger.error(f"Error en Prueba 1: {e}")

    # ------------------------------------------------------------------- #
    #  Pausa de seguridad entre pruebas (rate limit)
    # ------------------------------------------------------------------- #
    logger.info("\n--- ESPERANDO 2 SEGUNDOS ENTRE PRUEBAS (RATE LIMIT SAFETY) ---")
    await asyncio.sleep(2)

    # ------------------------------------------------------------------- #
    #  PRUEBA 2: Pregunta trampa (respuesta NO está en el contexto)
    # ------------------------------------------------------------------- #
    query_2 = "¿Cómo se configura un servidor Express.js con middleware de autenticación JWT?"

    logger.info("\n--- PRUEBA 2: Pregunta trampa (sin respuesta en el contexto) ---")
    logger.info(f"Consulta: \"{query_2}\"")

    try:
        result_2 = await get_rag_response(query_2, config=config)
        json_output_2 = json.dumps(result_2.model_dump(), indent=2, ensure_ascii=False)
        logger.info(f"RAGResponse validado (JSON):\n{json_output_2}")
        print("\n--- SALIDA JSON (PRUEBA 2 — Pregunta trampa, sin info en contexto) ---")
        print(json_output_2)
    except Exception as e:
        logger.error(f"Error en Prueba 2: {e}")

    logger.info("=== EJECUCIÓN DE PRUEBAS RAG FINALIZADA ===")


async def main():
    await run_rag_tests()


if __name__ == "__main__":
    asyncio.run(main())
