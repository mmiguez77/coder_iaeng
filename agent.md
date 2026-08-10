# Bitácora del Agente - Coderhouse AI Engineering

Este archivo documenta el progreso, las decisiones de diseño arquitectónico, las lecciones aprendidas y el control de cambios de la implementación del Módulo 1 en el directorio `app/`.

---

## 1. Módulo 1: Conexión y Abstracción de LLMs

### Decisiones de Diseño y Arquitectura (Clean Architecture)
- **Modularidad en Carpetas**: El proyecto está estructurado bajo una organización limpia separando responsabilidades:
  - `config/`: Configuración global y carga de variables de entorno con `pydantic-settings`.
  - `domain/entities/`: Definición de la interfaz abstracta e inmutable de los clientes (`BaseLLMClient`).
  - `domain/schemas/`: Contratos de datos de Pydantic V2 (`ChatMessage`, `LLMConfig`, `ModelResponse`).
  - `domain/exceptions/`: Excepciones personalizadas del dominio para la gestión centralizada de fallos.
  - `infrastructure/clients/`: Adaptadores concretos para interactuar con las APIs de los proveedores (`OpenAIClient`, `AnthropicClient` y `GeminiClient`).
  - `infrastructure/factory.py`: Implementación del *Factory Pattern* (`LLMFactory`).
  - `core/`: Orquestación y concurrencia asíncrona (`AsyncLLMManager`).
  - `logs/`: Directorio dedicado al almacenamiento físico de logs.
- **Formato de Logs Estricto**: Trazado configurado usando `loguru` sin ningún tipo de emojis ni iconos, mostrando la fecha y hora exacta en formato `DD/MM/YY - HH.mm.ss` (`%d/%m/%y - %H:%M:%S`).
- **Prueba e Integración de Gemini**:
  - Se eliminó el `MockLLMClient` para utilizar conexiones de red reales de extremo a extremo.
  - La única prueba implementada en `main.py` es la consulta de colores (listar 5 colores en inglés y español), demostrando tanto la ejecución multimodelo concurrente en caso de estar vacío `DEFAULT_PROVIDER` como la ejecución de streaming.
  - Las llaves de OpenAI y Anthropic son inválidas a propósito para forzar errores reales controlados de autenticación HTTP 401 en la red, mientras que Gemini (con la clave válida) resuelve exitosamente.

---

## 2. Registro de Tareas y Progreso

### Hitos
1. **[2026-08-10]** Creación e inicio del proyecto con la definición del Plan de Trabajo.
2. **[2026-08-10]** Reestructuración modular completa en carpetas siguiendo principios de arquitectura limpia.
3. **[2026-08-10]** Eliminación de Mocks y emojis, validación de la prueba única de colores y de la conmutación/concurrencia con fallos 401 y éxito real de Gemini.

---

## 3. Entorno de Ejecución y Dependencias
El proyecto requiere Python 3.12+ y las siguientes librerías instaladas en `app/.venv`:
- `openai`
- `anthropic`
- `google-genai`
- `pydantic`
- `pydantic-settings`
- `python-dotenv`
- `loguru`

---

## 4. Control de Cambios
| Fecha | Versión | Autor | Descripción del Cambio |
| :--- | :--- | :--- | :--- |
| 2026-08-10 | v0.1.0 | Antigravity | Inicio del repositorio y plan de trabajo original. |
| 2026-08-10 | v0.2.0 | Antigravity | Reestructuración a Clean Architecture, incorporación de Gemini, logs formateados y eliminación de MockLLMClient. |
| 2026-08-10 | v0.3.0 | Antigravity | Simplificación de la prueba a color_test únicamente, eliminación de todo tipo de emojis de código y logs. |
