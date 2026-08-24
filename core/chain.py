import asyncio
from typing import Optional
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from app.config.settings import load_settings
from app.domain.schemas.chat import LLMConfig, Provider
from app.domain.schemas.technical_entity import TechnicalEntityExtraction


def get_langchain_model(config: LLMConfig, model_name: Optional[str] = None) -> BaseChatModel:
    """
    Fábrica agnóstica para instanciar el cliente de chat de LangChain según la configuración.
    Soporta Google Gemini, OpenAI y Anthropic dinámicamente según el proveedor activo.
    """
    provider = config.default_provider

    # Si no se especificó un proveedor explícito, inferir a partir de las credenciales válidas disponibles
    if not provider:
        if config.gemini_api_key and not config.gemini_api_key.get_secret_value().startswith("invalid_"):
            provider = Provider.GEMINI
        elif config.openai_api_key and not config.openai_api_key.get_secret_value().startswith("invalid_"):
            provider = Provider.OPENAI
        elif config.anthropic_api_key and not config.anthropic_api_key.get_secret_value().startswith("invalid_"):
            provider = Provider.ANTHROPIC
        else:
            provider = Provider.GEMINI

    logger.debug(f"get_langchain_model: Proveedor seleccionado -> {provider.value}")

    if provider == Provider.GEMINI:
        if not config.gemini_api_key:
            raise ValueError("No se configuró gemini_api_key en las variables de entorno.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        target_model = model_name or config.default_model or "gemini-2.5-flash"
        logger.info(f"Instanciando ChatGoogleGenerativeAI con modelo: {target_model}")
        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=config.gemini_api_key.get_secret_value(),
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )

    elif provider == Provider.OPENAI:
        if not config.openai_api_key:
            raise ValueError("No se configuró openai_api_key en las variables de entorno.")
        from langchain_openai import ChatOpenAI
        target_model = model_name or "gpt-4o"
        logger.info(f"Instanciando ChatOpenAI con modelo: {target_model}")
        return ChatOpenAI(
            model=target_model,
            api_key=config.openai_api_key.get_secret_value(),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    elif provider == Provider.ANTHROPIC:
        if not config.anthropic_api_key:
            raise ValueError("No se configuró anthropic_api_key en las variables de entorno.")
        from langchain_anthropic import ChatAnthropic
        target_model = model_name or "claude-3-5-sonnet-20241022"
        logger.info(f"Instanciando ChatAnthropic con modelo: {target_model}")
        return ChatAnthropic(
            model=target_model,
            api_key=config.anthropic_api_key.get_secret_value(),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    else:
        raise ValueError(f"Proveedor '{provider}' no soportado.")


def build_extraction_chain(config: Optional[LLMConfig] = None):
    """
    Construye la cadena LCEL resiliente para la extracción de entidades técnicas.
    Composición: ChatPromptTemplate | Model.with_structured_output(TechnicalEntityExtraction).with_retry()
    """
    if config is None:
        config = load_settings()

    model = get_langchain_model(config)

    # Configuración de salida estructurada Pydantic mediante with_structured_output
    try:
        structured_llm = model.with_structured_output(TechnicalEntityExtraction, method="json_schema")
    except Exception:
        structured_llm = model.with_structured_output(TechnicalEntityExtraction)

    # Añadimos resiliencia con estrategia de reintentos automatizada (.with_retry)
    resilient_llm = structured_llm.with_retry(
        stop_after_attempt=5,
        wait_exponential_jitter=True
    )

    # Prompt Template modular utilizando ChatPromptTemplate con roles System y Human
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres un ingeniero de IA y arquitecto de software experto. "
            "Tu tarea es analizar el texto técnico y extraer la información estructurada solicitada. "
            "DEBES responder ÚNICAMENTE con el objeto JSON que cumpla el esquema."
        ),
        (
            "human",
            "Analiza el siguiente texto de entrada y extrae las entidades técnicas:\n\n{text}"
        )
    ])

    # Cadena LCEL declarativa compuesta: prompt | resilient_llm
    chain = prompt | resilient_llm
    return chain


async def process_text(text: str, config: Optional[LLMConfig] = None, max_retries: int = 3) -> TechnicalEntityExtraction:
    """
    Ejecuta la cadena LCEL de forma asíncrona usando .ainvoke().
    Incluye manejo proactivo de límites de tasa (429 Rate Limits / Quotas) con reintentos exponenciales.

    Args:
        text: El párrafo de texto técnico sin procesar.
        config: Configuración opcional del LLM (si es None, se carga del .env).
        max_retries: Intentos máximos en caso de límite de tasa de la API.

    Returns:
        Un objeto validado TechnicalEntityExtraction.
    """
    logger.info("Iniciando procesamiento asíncrono de texto técnico con la cadena LCEL...")
    logger.debug(f"Longitud del texto recibido: {len(text)} caracteres")

    if config is None:
        config = load_settings()

    chain = build_extraction_chain(config)
    logger.info("Cadena LCEL (Prompt + Structured Model + Retry) construida exitosamente. Ejecutando .ainvoke()...")

    for attempt in range(1, max_retries + 1):
        try:
            result: TechnicalEntityExtraction = await chain.ainvoke({"text": text})

            logger.info("Procesamiento y validación Pydantic completados exitosamente.")
            logger.info(f"Tecnologías extraídas: {result.tecnologias}")
            logger.info(f"Nivel de criticidad: {result.nivel_de_criticidad.value}")
            logger.info(f"Resumen técnico: {result.resumen_tecnico}")

            return result

        except Exception as e:
            err_str = str(e)
            is_rate_limit = any(k in err_str for k in ("429", "RESOURCE_EXHAUSTED", "Quota exceeded", "Rate limit"))
            
            if is_rate_limit and attempt < max_retries:
                wait_seconds = 25 * attempt
                logger.warning(
                    f"Límite de tasa detectado (429 Rate Limit/Quota). Reintentando en {wait_seconds}s (Intento {attempt}/{max_retries})..."
                )
                await asyncio.sleep(wait_seconds)
            elif is_rate_limit:
                clean_msg = (
                    "Límite de cuota alcanzado en la API del proveedor (429 RESOURCE_EXHAUSTED). "
                    "Por favor, espere 1 minuto a que se restablezca la cuota de la API antes de reintentar."
                )
                logger.error(clean_msg)
                raise RuntimeError(clean_msg) from e
            else:
                logger.error(f"Error irrecuperable durante la ejecución del pipeline de extracción: {e}")
                raise e
