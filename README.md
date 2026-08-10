# Coderhouse AI Engineering - Módulo 1: Conexión y Abstracción de LLMs

Este repositorio contiene la implementación del Módulo 1 del programa de AI Engineering, estructurado bajo los lineamientos de **Clean Architecture** (Arquitectura Limpia) y mejores prácticas en Python 3.12. 

El proyecto implementa un Unified Async LLM Client (cliente unificado asíncrono e intercambiable para OpenAI, Anthropic y Gemini) y un orquestador concurrente con control de flujo (semáforos) y resiliencia ante fallos (timeouts).

---

## Estructura Modular del Proyecto

Todo el código fuente y su configuración se encuentra aislado dentro del directorio `app/`:

```text
coderhouse/
├── README.md               # Este archivo de documentación
├── agent.md                # Bitácora del agente y control de cambios
└── app/
    ├── config/
    │   └── settings.py     # Carga de variables de entorno y configuración con pydantic-settings
    ├── domain/
    │   ├── entities/
    │   │   └── base_client.py # Interfaz abstracta común (BaseLLMClient)
    │   ├── exceptions/
    │   │   └── base.py     # Jerarquía de excepciones personalizadas para el dominio
    │   └── schemas/
    │       └── chat.py     # Modelos de datos para chat y enums (ChatMessage, LLMConfig, etc.)
    ├── infrastructure/
    │   ├── clients/
    │   │   ├── openai_client.py
    │   │   ├── anthropic_client.py
    │   │   └── gemini_client.py # Integración asíncrona con el SDK de google-genai
    │   └── factory.py      # Patrón Factory para la creación de clientes (LLMFactory)
    ├── core/
    │   └── manager.py      # Orquestador de llamadas concurrentes (AsyncLLMManager)
    ├── logs/
    │   └── execution.log   # Archivo de logs de ejecución formateados
    ├── .env                # Archivo de credenciales local (ignorado por git)
    ├── .env.example        # Plantilla de variables de entorno
    ├── .gitignore          # Exclusiones de Git locales del módulo
    └── main.py             # Punto de entrada principal y validador del sistema
```

---

## Requisitos Previos

- **Python**: Versión 3.12 o superior.
- **Librerías principales**:
  - `google-genai` (SDK oficial de Google)
  - `openai`
  - `anthropic`
  - `pydantic` y `pydantic-settings`
  - `python-dotenv`
  - `loguru` (para trazado y logs estructurados)

---

## Configuración y Variables de Entorno

1. Copia el archivo de plantilla a tu archivo local de variables de entorno:
   ```bash
   cp app/.env.example app/.env
   ```
2. Abre [`app/.env`](app/.env) y configura tus credenciales:
   - Ingresa tus API Keys reales de los proveedores.
   - **`DEFAULT_PROVIDER`**:
     - Establécelo con un proveedor específico (`openai`, `anthropic` o `gemini`) para realizar la consulta únicamente a ese modelo.
     - Déjalo **vacío** para indicar al orquestador concurrente que realice la consulta a los tres modelos en paralelo simultáneamente.

*Nota: Para verificar la tolerancia a fallos, puedes inyectar API Keys incorrectas de OpenAI y Anthropic, y la API Key real de Gemini. Verás cómo las dos primeras reportan errores de autenticación 401 reales capturados de forma segura, mientras que Gemini responde exitosamente y continúa la ejecución.*

---

## Instrucciones de Instalación y Ejecución

Sigue estos pasos desde la raíz del proyecto (`coderhouse/`):

### 1. Crear y Activar el Entorno Virtual
Crea el entorno virtual dentro de la carpeta `app/` para mantener el proyecto limpio y aislado:

```bash
# Crear entorno virtual
python3 -m venv app/.venv

# Activar el entorno virtual
source app/.venv/bin/activate
```

### 2. Instalar Dependencias
Instala los paquetes necesarios en el entorno virtual activo:

```bash
pip install -r app/.env.example  # o directamente las librerías:
pip install openai anthropic google-genai pydantic pydantic-settings python-dotenv loguru
```

### 3. Ejecutar la Aplicación
Para correr la prueba única de los colores (que evalúa la selección de proveedor, la concurrencia asíncrona mediante semáforos/timeouts y el streaming de tokens), ejecuta desde la raíz:

```bash
python -m app.main
```

Si deseas ejecutar desde la misma carpeta `app/`, debes configurar temporalmente el `PYTHONPATH`:

```bash
cd app
PYTHONPATH=.. python main.py
```

---

## Observabilidad y Logs

El sistema utiliza `loguru` para registrar la ejecución completa del flujo.
- **Formato de Fecha Estricto**: Todos los logs en consola y archivo usan la fecha y hora sin emojis en formato `dd/mm/yy - hh.mm.ss` (`%d/%m/%y - %H:%M:%S`).
- **Archivo de Logs**: Cada corrida del sistema se registra físicamente en [`app/logs/execution.log`](app/logs/execution.log).
