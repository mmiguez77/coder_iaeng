import asyncio
import sys
from pathlib import Path
from loguru import logger

from app.config.settings import load_settings
from app.core.manager import AsyncLLMManager
from app.domain.schemas.chat import ChatMessage, RoleEnum, Provider
from app.infrastructure.factory import LLMFactory

# Configuración de logs sin emojis y con formato dd/mm/yy - hh.mm.ss
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "execution.log"

logger.remove()  # Remover el logger por defecto de loguru

# Configurar el logger para consola
logger.add(
    sys.stderr,
    format="<green>{time:DD/MM/YY - HH.mm.ss}</green> | <level>{level: <8}</level> | {function}:{line} - {message}",
    level="INFO",
)

# Configurar el logger para el archivo físico app/logs/execution.log
logger.add(
    str(log_file_path),
    format="{time:DD/MM/YY - HH.mm.ss} | {level: <8} | {function}:{line} - {message}",
    level="INFO",
    rotation="10 MB",
)


async def execute_color_test(config):
    """
    Realiza la prueba de colores según la configuración del archivo .env.
    Si default_provider está configurado, consulta solo a ese proveedor.
    Si está vacío, consulta concurrentemente a los tres proveedores.
    """
    logger.info("=== INICIANDO PRUEBA DE COLORES ===")
    
    manager = AsyncLLMManager(concurrency_limit=2, default_timeout=5.0)

    # Mensaje de prueba unificado
    messages = [
        ChatMessage(
            role=RoleEnum.USER,
            content="Enumera 5 colores en ingles y espanol."
        )
    ]

    # Ejecutamos las llamadas usando el manager
    results = await manager.generate_from_config(messages, config)

    logger.info("=== RESULTADOS DE LAS CONSULTAS (MODO NORMAL) ===")
    
    for res in results:
        prov = res.provider.value.upper()
        if res.success:
            logger.info(f"[SUCCESS] [{prov}] Respuesta del modelo ({res.model}):\n{res.content.strip()}")
        else:
            logger.error(f"[ERROR] [{prov}] Error en la consulta ({res.model}): {res.error}")

    # Demostración de streaming para los proveedores que hayan respondido exitosamente
    logger.info("=== INICIANDO PRUEBA DE STREAMING ===")
    for res in results:
        if res.success:
            prov = res.provider
            logger.info(f"Iniciando stream para: {prov.value.upper()}")
            try:
                client = LLMFactory.create_client(prov, config)
                print(f"Stream [{prov.value.upper()}]: ", end="", flush=True)
                async for token in client.generate_stream(messages):
                    print(token, end="", flush=True)
                print("\n")
            except Exception as e:
                logger.error(f"Fallo en streaming para {prov.value.upper()}: {e}")

    logger.info("=== FIN DE LAS PRUEBAS ===")


async def main():
    logger.info("Iniciando la aplicacion del Modulo 1...")
    
    try:
        config = load_settings()
    except Exception as e:
        logger.critical(f"Error critico al cargar las configuraciones: {e}")
        sys.exit(1)

    # Ejecutar la prueba de colores
    await execute_color_test(config)


if __name__ == "__main__":
    asyncio.run(main())
