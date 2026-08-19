# Coderhouse AI Engineering - Módulo 2: Pipeline de Procesamiento Validado con LCEL

Este repositorio contiene la implementación del **Módulo 2 (Pre-entrega 2)** del programa AI Engineering: **Pipeline de Extracción de Entidades Técnicas**. 

El sistema implementa una arquitectura declarativa no bloqueante mediante **LangChain Expression Language (LCEL)**, un contrato de datos estricto con **Pydantic v2** (`with_structured_output`), resiliencia y reintentos automáticos ante fallos de formato o red (`with_retry`), y un diseño 100% agnóstico respecto al proveedor de modelos de lenguaje (LLM).

---

## Características Principales

1. **Diseño 100% Agnóstico al Proveedor**:
   - Soporta **OpenAI**, **Anthropic** y **Google Gemini** de forma transparente.
   - El proveedor (`DEFAULT_PROVIDER`), el modelo (`DEFAULT_MODEL`) y sus credenciales se configuran exclusivamente desde las variables de entorno (`.env`).
   - Una fábrica agnóstica (`get_langchain_model`) se encarga de instanciar la clase de chat correspondiente de LangChain (`ChatOpenAI`, `ChatAnthropic` o `ChatGoogleGenerativeAI`).

2. **Esquema Pydantic Estricto (`schemas.py`)**:
   - `tecnologias` (`list[str]`): Lista de tecnologías o herramientas identificadas (mínimo 1 elemento, no vacía).
   - `nivel_de_criticidad` (`CriticidadEnum`): Criticidad del evento/sistema (`baja`, `media`, `alta`).
   - `resumen_tecnico` (`str`): Resumen del problema, arquitectura o estado descrito.

3. **Cadena LCEL y Resiliencia (`chain.py`)**:
   - Cadena declarativa compuesta mediante el operador pipe (`|`):
     ```python
     chain = prompt | model.with_structured_output(TechnicalEntityExtraction).with_retry(stop_after_attempt=5)
     ```
   - Invocación no bloqueante con `.ainvoke()`.
   - Reintentos exponenciales automáticos con `.with_retry()` para recuperación ante errores de red o cuotas de consumo.

4. **Trazabilidad y Logging**:
   - Integración con `loguru` para el registro estructurado de eventos en consola y en el archivo físico [`app/logs/execution.log`](app/logs/execution.log).

---

## Estructura del Proyecto

```text
coderhouse/
├── schemas.py              # Re-exportación raíz de esquemas Pydantic del dominio
├── chain.py                # Re-exportación raíz del pipeline LCEL y función process_text
├── app/
│   ├── config/
│   │   └── settings.py     # Carga y validación de variables de entorno (LLMConfig)
│   ├── domain/
│   │   ├── entities/
│   │   │   └── base_client.py
│   │   └── schemas/
│   │       ├── chat.py             # Esquemas de configuración y chat del Módulo 1
│   │       └── technical_entity.py # Esquema Pydantic del Módulo 2 (TechnicalEntityExtraction)
│   ├── infrastructure/
│   │   ├── clients/
│   │   │   ├── openai_client.py
│   │   │   ├── anthropic_client.py
│   │   │   └── gemini_client.py
│   │   └── factory.py
│   ├── core/
│   │   ├── manager.py
│   │   └── chain.py        # Construcción de la cadena LCEL y función asíncrona process_text
│   ├── logs/
│   │   └── execution.log   # Registros de ejecución formateados
│   ├── .env                # Variables de entorno locales (ignorado en git)
│   ├── .env.example        # Plantilla de variables de entorno
│   ├── requirements.txt    # Dependencias del proyecto
│   ├── main.py             # Script de prueba asíncrono del pipeline
│   ├── schemas.py          # Acceso modular a los esquemas
│   ├── chain.py            # Acceso modular a la cadena
│   └── README.md           # Documentación del proyecto
```

---

## Configuración y Variables de Entorno

1. Copiar la plantilla `.env.example` a `.env` dentro de la carpeta `app/`:
   ```bash
   cp app/.env.example app/.env
   ```
2. Configurar las variables en `.env`:
   - `DEFAULT_PROVIDER`: `openai`, `anthropic` o `gemini` (o dejar vacío para detección automática según credenciales válidas).
   - `DEFAULT_MODEL`: Nombre del modelo deseado (ej: `gpt-4o`, `claude-3-5-sonnet-20241022`, `gemini-2.5-flash`).
   - Credenciales: Ingresar la API Key correspondiente al proveedor seleccionado (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, o `GEMINI_API_KEY`).

---

## Instalación y Ejecución

### 1. Activar el Entorno Virtual
```bash
source app/.venv/bin/activate
```

### 2. Instalar Dependencias
```bash
pip install -r app/requirements.txt
```

### 3. Ejecutar el Script de Prueba Asíncrono
Desde la raíz del repositorio (`coderhouse/`):

```bash
PYTHONPATH=. python -m app.main
```

O navegando dentro del directorio `app/`:

```bash
cd app
PYTHONPATH=.. python main.py
```

---

## Ejemplo de Salida del Pipeline

Dada una entrada de texto técnico (log de error o descripción de infraestructura), la función `process_text(text: str)` ejecuta la cadena LCEL de forma asíncrona y devuelve un objeto validado por Pydantic como este:

```json
{
  "tecnologias": [
    "FastAPI",
    "Python 3.12",
    "PostgreSQL",
    "Redis"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El servicio desarrollado en FastAPI y Python 3.12 sufrió una degradación crítica debido a un cuello de botella en PostgreSQL provocado por la caída del clúster de caché en Redis."
}
```
