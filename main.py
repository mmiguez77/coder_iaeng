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
from app.core.chain import process_text

# Configuración de logs con formato dd/mm/yy - hh.mm.ss en consola y archivo
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "execution.log"

logger.remove()  # Remover el logger por defecto de loguru

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


async def run_pipeline_tests():
    """
    Ejecuta casos de prueba sobre la cadena LCEL con salida estructurada y resiliencia.
    """
    logger.info("=== INICIANDO PRUEBAS DEL PIPELINE LCEL (MODULO 2) ===")

    try:
        config = load_settings()
        logger.info("Configuración cargada correctamente desde las variables de entorno.")
    except Exception as e:
        logger.critical(f"Error al cargar las configuraciones: {e}")
        sys.exit(1)

    # Caso de prueba 1: Descripción de Arquitectura y Error de Backend
    sample_text_1 = (
        "El servicio web desarrollado con FastAPI y Python 3.12 presentó una degradación crítica. "
        "Se identificó un cuello de botella de conexiones concurrentes hacia la base de datos PostgreSQL "
        "debido a un fallo masivo en el clúster de caché con Redis. La latencia superó los 15 segundos y afectó al 80% de los usuarios."
    )

    logger.info("\n--- CASO DE PRUEBA 1: Log de Error e Infraestructura ---")
    logger.info(f"Texto de entrada:\n\"{sample_text_1}\"")
    
    result_1 = await process_text(sample_text_1, config)

    # Convertir a JSON formateado para verificación
    json_output_1 = json.dumps(result_1.model_dump(), indent=2, ensure_ascii=False)
    logger.info(f"Objeto Pydantic Validado (JSON):\n{json_output_1}")
    print("\n--- SALIDA ESPERADA JSON (CASO 1) ---")
    print(json_output_1)

    # Caso de prueba 2: Prueba de Estrés / Texto Ambiguo
    logger.info("\n--- ESPERANDO 2 SEGUNDOS ENTRE PRUEBAS (RATE LIMIT SAFETY) ---")
    await asyncio.sleep(2)

    sample_text_2 = (
        "Reporte no estructurado: Notamos problemas de red usando Docker, Kubernetes y un servicio en Go. "
        "Puede que la carga sea media pero hay inconsistencias en los pods."
    )

    logger.info("\n--- CASO DE PRUEBA 2: Texto Ambiguo (Prueba de Estrés) ---")
    logger.info(f"Texto de entrada:\n\"{sample_text_2}\"")

    result_2 = await process_text(sample_text_2, config)

    json_output_2 = json.dumps(result_2.model_dump(), indent=2, ensure_ascii=False)
    logger.info(f"Objeto Pydantic Validado (JSON):\n{json_output_2}")
    print("\n--- SALIDA ESPERADA JSON (CASO 2) ---")
    print(json_output_2)

    logger.info("=== PRUEBAS COMPLETADAS CON EXITO ===")


async def main():
    await run_pipeline_tests()


if __name__ == "__main__":
    asyncio.run(main())
