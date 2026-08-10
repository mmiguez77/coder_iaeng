# Coderhouse AI Engineering - Módulo 1: Conexión y Abstracción de LLMs

Este repositorio contiene los desafíos del curso de AI Engineering. 
El proyecto implementa un Unified Async LLM Client (cliente unificado asíncrono e intercambiable para OpenAI, Anthropic y Gemini) y un orquestador concurrente con control de flujo (semáforos) y resiliencia ante fallos (timeouts).

---

## Estructura Modular del Proyecto

Todo el código fuente y su configuración se encuentran aislados dentro del directorio `app/`:

```text
app/
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
  ├── main.py             # Punto de entrada principal y validador del sistema
  └── README.md           # Este archivo de documentación
```

---

## Requisitos Previos

- **Python**: Versión 3.12 o superior.
- **Librerías principales**:
  - `google-genai`
  - `openai`
  - `anthropic`
  - `pydantic` y `pydantic-settings`
  - `python-dotenv`
  - `loguru`

---

## Configuración y Variables de Entorno

1. Copiar y renombrar el archivo de plantilla `.env.example` a `.env` dentro de la carpeta `app/`:
   ```bash
   cp app/.env.example app/.env
   ```
2. Modificar las variables del archivo `.env` resultante e ingresar tus API Keys de OpenAI, Anthropic y Gemini.
3. Configurar el proveedor por defecto en la variable `DEFAULT_PROVIDER`:
   - Completar para trabajar con un proveedor específico (`openai`, `anthropic` o `gemini`).
   - Dejar vacío para indicar al orquestador que realice la consulta a los tres modelos en paralelo simultáneamente.

---

## Instrucciones de Instalación y Ejecución

Pasos para la instalación y ejecución desde la raíz del proyecto (`coderhouse/`):

### 1. Crear y Activar el Entorno Virtual
Para mantener el proyecto limpio y aislado, crear el entorno virtual dentro de la carpeta `app/`:

```bash
# Crear entorno virtual
python3 -m venv app/.venv

# Activar el entorno virtual
source app/.venv/bin/activate
```

### 2. Instalar Dependencias
Instalar los paquetes necesarios en el entorno virtual activo:

```bash
pip install -r app/.env.example  # o directamente las librerías:
pip install openai anthropic google-genai pydantic pydantic-settings python-dotenv loguru
```

### 3. Ejecutar la Aplicación
Para correr la prueba única (que evalúa la selección de proveedor, la concurrencia asíncrona mediante semáforos/timeouts y el streaming de tokens), ejecutar desde la raíz:

```bash
python -m app.main
```

Para ejecutar dentro de la carpeta, `app/`, configurar temporalmente `PYTHONPATH`:

```bash
cd app
PYTHONPATH=.. python main.py
```

---

## Observabilidad y Logs

El sistema utiliza `loguru` para registrar la ejecución completa del flujo.
- **Formato de Fecha Estricto**: Todos los logs en consola y archivo usan la fecha y hora sin emojis en formato `dd/mm/yy - hh.mm.ss` (`%d/%m/%y - %H:%M:%S`).
- **Archivo de Logs**: Cada corrida del sistema se registra físicamente en [`app/logs/execution.log`](app/logs/execution.log).
