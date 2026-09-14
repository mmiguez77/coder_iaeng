# Introducción a LangChain

## ¿Qué es LangChain?

LangChain es un framework de código abierto diseñado para facilitar el desarrollo de aplicaciones impulsadas por modelos de lenguaje (LLMs). Proporciona una capa de abstracción que permite conectar modelos de IA con fuentes de datos externas, herramientas y flujos de trabajo complejos de manera declarativa y componible.

## Componentes principales

LangChain se organiza en varios módulos fundamentales:

- **Models**: Interfaces unificadas para interactuar con distintos proveedores de LLMs como OpenAI, Anthropic, Google Gemini y modelos open-source de HuggingFace. Cada proveedor se abstrae detrás de una interfaz común (`BaseChatModel`), lo que permite cambiar de modelo sin modificar el código de la aplicación.

- **Prompts**: Sistema de plantillas (`ChatPromptTemplate`) que permite definir instrucciones reutilizables con variables dinámicas. Soporta roles de sistema, usuario y asistente para controlar el comportamiento del modelo.

- **Chains**: Secuencias de operaciones encadenadas donde la salida de un componente alimenta la entrada del siguiente. Permiten construir flujos de procesamiento complejos de forma modular.

- **Memory**: Mecanismos para mantener contexto entre interacciones, esencial para aplicaciones conversacionales que necesitan recordar el historial del diálogo.

## LangChain Expression Language (LCEL)

LCEL es el paradigma declarativo de LangChain para componer cadenas usando el operador pipe (`|`). En lugar de programar imperativamente cada paso, LCEL permite expresar un pipeline completo en una sola línea:

```python
chain = prompt | model | output_parser
```

Las ventajas principales de LCEL son:

1. **Composición declarativa**: El flujo se lee de izquierda a derecha, facilitando la comprensión.
2. **Soporte nativo asíncrono**: Cada componente expone métodos `.ainvoke()`, `.abatch()` y `.astream()` para ejecución no bloqueante.
3. **Resiliencia integrada**: Se puede agregar `.with_retry()` a cualquier componente para reintentos automáticos con backoff exponencial ante errores transitorios.
4. **Streaming**: Soporta transmisión token a token para respuestas en tiempo real.

## Ejemplo de cadena LCEL

Un pipeline típico combina un prompt template, un modelo con salida estructurada y reintentos:

```python
chain = prompt | model.with_structured_output(Schema).with_retry(stop_after_attempt=3)
result = await chain.ainvoke({"variable": "valor"})
```

Este patrón garantiza que la salida cumpla con un esquema Pydantic definido, y que los errores transitorios de la API se manejen automáticamente sin intervención del desarrollador.
