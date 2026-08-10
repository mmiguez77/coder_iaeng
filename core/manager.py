import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

from app.domain.entities.base_client import BaseLLMClient
from app.domain.schemas.chat import ChatMessage, LLMConfig, ModelResponse, Provider
from app.infrastructure.factory import LLMFactory


class AsyncLLMManager:
    """Orquestador para llamadas asíncronas concurrentes a múltiples modelos con control de flujo y tiempos."""

    def __init__(self, concurrency_limit: int = 2, default_timeout: float = 5.0):
        """
        Inicializa el manager.

        Args:
            concurrency_limit: Límite de tareas simultáneas ejecutándose en el semáforo.
            default_timeout: Tiempo límite para la ejecución de cada llamada en segundos.
        """
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.default_timeout = default_timeout
        logger.info(
            f"AsyncLLMManager: Concurrencia limitada a {concurrency_limit}, Timeout por defecto = {default_timeout}s"
        )

    async def execute_single_client(
        self,
        client: BaseLLMClient,
        messages: List[ChatMessage],
        timeout_sec: float
    ) -> ModelResponse:
        """
        Ejecuta la llamada de un cliente bajo el semáforo y el timeout correspondientes.
        Mapea errores a ModelResponse con error asignado para evitar crashes.
        """
        provider_name = client.__class__.__name__.replace("Client", "").upper()
        logger.debug(f"[{provider_name}] Esperando cupo en el semáforo...")
        
        async with self.semaphore:
            logger.info(f"[{provider_name}] Semáforo adquirido. Iniciando llamada...")
            start_time = asyncio.get_event_loop().time()
            
            try:
                async with asyncio.timeout(timeout_sec):
                    # Invocamos al cliente. El cliente internamente captura sus excepciones y retorna un ModelResponse.
                    response = await client.generate(messages)
                
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.info(f"[{provider_name}] Respuesta obtenida con éxito en {elapsed:.2f}s")
                return response
                
            except TimeoutError:
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.warning(f"[{provider_name}] Timeout excedido tras {elapsed:.2f}s")
                # Intentamos mapear al proveedor correcto de forma dinámica
                provider_enum = Provider.GEMINI
                if "OPENAI" in provider_name:
                    provider_enum = Provider.OPENAI
                elif "ANTHROPIC" in provider_name:
                    provider_enum = Provider.ANTHROPIC
                    
                return ModelResponse(
                    provider=provider_enum,
                    model=client.model_name,
                    content="",
                    error=f"Timeout de ejecución excedido ({timeout_sec}s).",
                    success=False
                )
            except Exception as e:
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.error(f"[{provider_name}] Error crítico e inesperado en la llamada: {e}")
                provider_enum = Provider.GEMINI
                if "OPENAI" in provider_name:
                    provider_enum = Provider.OPENAI
                elif "ANTHROPIC" in provider_name:
                    provider_enum = Provider.ANTHROPIC

                return ModelResponse(
                    provider=provider_enum,
                    model=client.model_name,
                    content="",
                    error=f"Error del sistema: {str(e)}",
                    success=False
                )

    async def generate_from_config(
        self,
        messages: List[ChatMessage],
        config: LLMConfig,
        timeout_sec: Optional[float] = None
    ) -> List[ModelResponse]:
        """
        Ejecuta las consultas según la configuración.
        - Si config.default_provider está definido: Consulta únicamente a ese proveedor.
        - Si config.default_provider está vacío (o None): Consulta concurrentemente a los tres proveedores (OpenAI, Anthropic, Gemini).

        Args:
            messages: Lista de mensajes de chat.
            config: Configuración de proveedores y credenciales.
            timeout_sec: Timeout específico.

        Returns:
            Lista de ModelResponse obtenidos.
        """
        timeout_val = timeout_sec if timeout_sec is not None else self.default_timeout
        provider = config.default_provider

        # Caso 1: Se especificó un único proveedor a utilizar
        if provider is not None:
            logger.info(f"Proveedor único especificado en configuración: '{provider.value}'")
            try:
                client = LLMFactory.create_client(provider, config)
                res = await self.execute_single_client(client, messages, timeout_val)
                return [res]
            except Exception as e:
                logger.error(f"Error al inicializar cliente para '{provider.value}': {e}")
                return [ModelResponse(
                    provider=provider,
                    model=config.default_model,
                    content="",
                    error=f"Fallo de inicialización del cliente: {str(e)}",
                    success=False
                )]

        # Caso 2: El proveedor está vacío. Realizamos la consulta concurrente a los 3 proveedores.
        logger.info("No se especificó proveedor por defecto. Iniciando consulta concurrente a los tres modelos...")
        
        # Intentamos instanciar los clientes correspondientes a cada proveedor
        providers = [Provider.OPENAI, Provider.ANTHROPIC, Provider.GEMINI]
        tasks = []
        
        for p in providers:
            try:
                # Cada proveedor puede tener su modelo específico según la convención
                model_name = None
                if p == Provider.GEMINI:
                    model_name = "gemini-2.5-flash"
                elif p == Provider.ANTHROPIC:
                    model_name = "claude-3-5-sonnet-20241022"
                elif p == Provider.OPENAI:
                    model_name = "gpt-4o-mini"
                    
                client = LLMFactory.create_client(p, config, model_name=model_name)
                # Encapsulamos la corrutina de llamada asíncrona
                tasks.append(self.execute_single_client(client, messages, timeout_val))
            except Exception as e:
                logger.error(f"Fallo de instanciación para el proveedor '{p.value}': {e}")
                # Creamos una tarea dummy que devuelva el error directamente para no romper el gather
                async def mock_error_task(prov=p, err=e) -> ModelResponse:
                    return ModelResponse(
                        provider=prov,
                        model="unknown",
                        content="",
                        error=f"Fallo al crear cliente: {str(err)}",
                        success=False
                    )
                tasks.append(mock_error_task())

        # Ejecutamos las llamadas concurrentemente usando gather
        results = await asyncio.gather(*tasks)
        return list(results)
