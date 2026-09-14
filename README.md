# Coderhouse AI Engineering — Pipeline LCEL + RAG Local + RAG Cloud Híbrido

Este repositorio contiene la implementación de los módulos del programa **AI Engineering de Coderhouse**:

- **Módulo 2**: Pipeline de Extracción de Entidades Técnicas (LCEL + Pydantic)
- **Módulo 3**: Sistema de Recuperación Semántica Local (RAG End-to-End con ChromaDB)
- **Módulo 4**: Sistema RAG Escalable en la Nube (Pinecone Serverless + Recuperación Híbrida BM25 + Evaluación Cuantitativa)

---

## Tabla de Contenidos

1. [Características Principales](#características-principales)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Configuración y Variables de Entorno](#configuración-y-variables-de-entorno)
4. [Instalación](#instalación)
5. [Módulo 2 — Pipeline de Extracción](#módulo-2--pipeline-de-extracción-de-entidades-técnicas)
6. [Módulo 3 — Sistema RAG Local](#módulo-3--sistema-rag-local)
7. [Módulo 4 — Sistema RAG Escalable en Pinecone con Búsqueda Híbrida](#módulo-4--sistema-rag-escalable-en-pinecone-con-búsqueda-híbrida)
8. [Resultados de Evaluación y Métricas](#resultados-de-evaluación-del-módulo-4)

---

## Características Principales

### Diseño Agnóstico al Proveedor
- Soporta **OpenAI**, **Anthropic** y **Google Gemini** de forma transparente.
- El proveedor y modelo se configuran desde variables de entorno (`.env`).
- Una fábrica agnóstica (`get_langchain_model`) instancia el cliente de chat correspondiente.

### Módulo 2 — Extracción LCEL
- Cadena LCEL: `prompt | model.with_structured_output(Schema).with_retry()`
- Esquema Pydantic estricto (`TechnicalEntityExtraction`) con validaciones.
- Invocación no bloqueante con `.ainvoke()` y reintentos exponenciales.

### Módulo 3 — RAG End-to-End
- **Ingesta**: Chunking basado en tokens (`RecursiveCharacterTextSplitter.from_tiktoken_encoder`) con persistencia en ChromaDB.
- **Retriever**: Búsqueda de similitud con embeddings locales y gratuitos (`HuggingFace sentence-transformers`).
- **Generación Grounded**: Cadena LCEL con `PydanticOutputParser` y prompt con filtro de veracidad.
- **Anti-alucinación**: El modelo rechaza preguntas fuera del contexto con: *"No tengo acceso a esa información en los documentos disponibles."*
- **Fuentes verificables**: Las referencias provienen de los metadatos reales de ChromaDB, no de la generación del LLM.

### Trazabilidad y Logging
- Integración con `loguru` para registro estructurado en consola y archivo (`logs/execution.log`).
- Medición de tiempo de ejecución por etapa (recuperación, generación, total).

---

## Estructura del Proyecto

```text
coderhouse/
├── app/
│   ├── config/
│   │   └── settings.py               # Carga y validación de variables de entorno (Pydantic)
│   ├── domain/
│   │   └── schemas/
│   │       ├── chat.py                # Esquemas de configuración y chat (M1/M2)
│   │       ├── technical_entity.py    # Esquema Pydantic del Módulo 2
│   │       ├── rag_response.py        # Esquemas Pydantic del Módulo 3
│   │       └── evaluation.py          # Esquemas Pydantic del Módulo 4 (Benchmark & Métricas)
│   ├── infrastructure/
│   │   ├── clients/
│   │   │   ├── openai_client.py
│   │   │   ├── anthropic_client.py
│   │   │   └── gemini_client.py
│   │   ├── factory.py                 # Fábrica de clientes LLM
│   │   ├── vector_store.py            # Gestión de ChromaDB local (Módulo 3)
│   │   └── pinecone_store.py          # Gestión de Pinecone Serverless Cloud (Módulo 4)
│   ├── core/
│   │   ├── chain.py                   # Cadena LCEL del Módulo 2 (extracción de entidades)
│   │   ├── rag_chain.py               # Cadena LCEL del Módulo 3 (RAG local + grounded prompt)
│   │   └── rag_system.py              # Clase RAGSystem híbrida BM25 + Pinecone (Módulo 4)
│   ├── data/
│   │   ├── langchain_intro.md         # Dataset técnico: Introducción a LangChain
│   │   ├── rag_concepts.md            # Dataset técnico: Conceptos de RAG
│   │   ├── pydantic_guide.md          # Dataset técnico: Guía de Pydantic v2
│   │   ├── chromadb_guide.md          # Dataset técnico: Guía de ChromaDB
│   │   ├── pinecone_serverless_guide.md # Dataset técnico: Pinecone Serverless y Búsqueda Híbrida
│   │   └── golden_set.json            # Benchmark Golden Set de 5 casos canónicos de prueba
│   ├── vectorstore/                   # Base vectorial ChromaDB local (gitignored)
│   ├── logs/
│   │   ├── execution.log              # Registros de ejecución estructurados (gitignored)
│   │   └── evaluation_report.json     # Reporte exportado de la evaluación cuantitativa
│   ├── main.py                        # Script de prueba del pipeline LCEL (Módulo 2)
│   ├── ingest.py                      # Script de ingesta local a ChromaDB (Módulo 3)
│   ├── rag.py                         # Script de prueba del sistema RAG local (Módulo 3)
│   ├── setup_pinecone.py              # Script de inicialización de Pinecone Serverless (Módulo 4)
│   ├── ingest_pinecone.py             # Script de ingesta inteligente a Pinecone Cloud (Módulo 4)
│   ├── evaluate.py                    # Script de evaluación cuantitativa Recall@5 y Precision@5 (Módulo 4)
│   ├── .env                           # Variables de entorno locales (gitignored)
│   ├── .env.example                   # Plantilla de variables de entorno con Pinecone
│   └── requirements.txt               # Dependencias del proyecto
```

---

## Configuración y Variables de Entorno

1. Copiar la plantilla `.env.example` a `.env` dentro de la carpeta `app/`:
   ```bash
   cp app/.env.example app/.env
   ```
2. Configurar las variables en `.env`:

   | Variable | Descripción | Ejemplo |
   |----------|-------------|---------|
   | `DEFAULT_PROVIDER` | Proveedor de LLM (`openai`, `anthropic`, `gemini` o vacío) | `gemini` |
   | `DEFAULT_MODEL` | Nombre del modelo | `gemini-2.5-flash` |
   | `OPENAI_API_KEY` | API Key de OpenAI (si aplica) | `sk-...` |
   | `ANTHROPIC_API_KEY` | API Key de Anthropic (si aplica) | `sk-ant-...` |
   | `GEMINI_API_KEY` | API Key de Google Gemini (si aplica) | `AQ...` |
   | `TEMPERATURE` | Temperatura del modelo (0.0 – 2.0) | `0.3` |
   | `MAX_TOKENS` | Tokens máximos de salida | `256` |
   | `EMBEDDINGS_MODEL` | Modelo de embeddings para RAG local (M3) | `sentence-transformers/all-MiniLM-L6-v2` |
   | `PINECONE_API_KEY` | API Key de Pinecone Serverless (M4) | `pcsk_...` |
   | `PINECONE_INDEX_NAME` | Nombre del índice en Pinecone (M4) | `coderhouse-rag-index` |
   | `PINECONE_CLOUD` | Proveedor Cloud Serverless (M4) | `aws` |
   | `PINECONE_REGION` | Región Cloud Serverless (M4) | `us-east-1` |
   | `PINECONE_NAMESPACE` | Namespace para segmentación lógica (M4) | `dev-technical-docs` |
   | `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings para Pinecone (M4) | `text-embedding-3-small` |

---

## Instalación

### 1. Crear y activar el entorno virtual
```bash
python -m venv app/.venv
source app/.venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r app/requirements.txt
```

---

## Módulo 2 — Pipeline de Extracción de Entidades Técnicas

### Descripción
Pipeline declarativo LCEL que analiza textos técnicos y extrae entidades estructuradas (tecnologías, criticidad, resumen) validadas con Pydantic v2.

### Ejecución
Desde la raíz del repositorio (`coderhouse/`):
```bash
PYTHONPATH=. python -m app.main
```

### Ejemplo de Salida
```json
{
  "tecnologias": ["FastAPI", "Python 3.12", "PostgreSQL", "Redis"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El servicio desarrollado en FastAPI y Python 3.12 sufrió una degradación crítica debido a un cuello de botella en PostgreSQL provocado por la caída del clúster de caché en Redis."
}
```

---

## Módulo 3 — Sistema RAG Local

### Descripción
Sistema de Recuperación Semántica (RAG) End-to-End que:
1. **Ingesta** documentos `.md` fragmentándolos en tokens y persistiéndolos en ChromaDB.
2. **Recupera** los fragmentos más relevantes ante una consulta del usuario.
3. **Genera** respuestas fundamentadas exclusivamente en el contexto recuperado.

### Componentes clave

| Componente | Archivo | Descripción |
|------------|---------|-------------|
| **Ingesta** | `ingest.py` | Carga, fragmenta y persiste documentos en ChromaDB |
| **Retriever** | `infrastructure/vector_store.py` | Embeddings locales HuggingFace + ChromaDB |
| **Cadena RAG** | `core/rag_chain.py` | LCEL: `prompt \| model \| PydanticOutputParser` |
| **Schema** | `domain/schemas/rag_response.py` | `RespuestaLLM` (output LLM) + `RAGResponse` (final) |
| **Pruebas** | `rag.py` | 2 casos: pregunta con contexto + pregunta trampa |
| **Dataset** | `data/*.md` | 4 documentos técnicos sobre LangChain, RAG, Pydantic, ChromaDB |

### Ejecución

#### Paso 1: Ingesta de documentos
```bash
PYTHONPATH=. python app/ingest.py
```

Salida esperada:
```
=== INICIANDO INGESTA DE DOCUMENTOS (MÓDULO 3 — RAG) ===
Documentos cargados: 4
Fragmentos generados: 11 (chunking en 0.12s)
Indexación completada en 0.27s
=== INGESTA FINALIZADA — Tiempo total: 18.88s ===
```

> **Nota:** La primera ejecución descarga el modelo de embeddings (~90MB). Las siguientes ejecuciones detectan el índice existente y omiten la re-indexación.

#### Paso 2: Pruebas del sistema RAG
```bash
PYTHONPATH=. python app/rag.py
```

### Ejemplo de Salida — Pregunta con respuesta en el contexto

Consulta: *"¿Qué es LCEL en LangChain y cómo se compone una cadena?"*

```json
{
  "respuesta": "LCEL (LangChain Expression Language) es el paradigma declarativo de LangChain que permite componer cadenas utilizando el operador pipe (`|`). En lugar de programar cada paso de forma imperativa, LCEL permite expresar un pipeline completo en una sola línea. Una cadena típica se compone conectando componentes como un prompt, un modelo y un parser de salida, por ejemplo: `chain = prompt | model | output_parser`.",
  "fuentes": [
    "data/langchain_intro.md",
    "data/pydantic_guide.md",
    "data/rag_concepts.md"
  ],
  "fragmentos_recuperados": 4
}
```

### Ejemplo de Salida — Pregunta trampa (sin respuesta en el contexto)

Consulta: *"¿Cómo se configura un servidor Express.js con middleware de autenticación JWT?"*

```json
{
  "respuesta": "No tengo acceso a esa información en los documentos disponibles.",
  "fuentes": [
    "data/chromadb_guide.md",
    "data/langchain_intro.md",
    "data/rag_concepts.md"
  ],
  "fragmentos_recuperados": 4
}
```

> El modelo identifica correctamente que la información sobre Express.js/JWT no está en el contexto y responde sin alucinar.

### Detalles técnicos

- **Chunking**: `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=70)` — basado en tokens reales, no caracteres.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, local y gratuito). El mismo modelo se usa para indexar y consultar.
- **top_k**: 4 fragmentos por consulta (rango recomendado: 3–5).
- **Persistencia**: ChromaDB con cliente persistente en `./vectorstore`. Verificación automática anti-re-indexación.
- **Schema en 2 niveles**: `RespuestaLLM` (lo que parsea el LLM, simple) + `RAGResponse` (objeto final con fuentes verificables de metadata real, no generadas por el modelo).

---

## Módulo 4 — Sistema RAG Escalable en Pinecone con Búsqueda Híbrida

### Descripción de la Arquitectura

El **Módulo 4** escala la recuperación semántica a una infraestructura empresarial en la nube con **Pinecone Serverless**, combinando recuperación vectorial densa con recuperación léxica esparsa (**BM25**) mediante un **Recuperador Híbrido** (`EnsembleRetriever`), y validando su rendimiento con métricas cuantitativas (**Recall@5**, **Precision@5**, **Hit Rate@5** y **MRR**).

```text
                                        ┌────────────────────────┐
                                        │  Consulta del Usuario  │
                                        └───────────┬────────────┘
                                                    │
                               ┌────────────────────┴────────────────────┐
                               │                                         │
                               ▼                                         ▼
                 ┌───────────────────────────┐             ┌───────────────────────────┐
                 │    Búsqueda Léxica        │             │    Búsqueda Semántica     │
                 │      BM25Retriever        │             │   Pinecone Serverless     │
                 │ (Términos exactos/código) │             │  (Similitud Coseno Cloud) │
                 └─────────────┬─────────────┘             └─────────────┬─────────────┘
                               │                                         │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                                     ┌─────────────────────────────┐
                                     │      EnsembleRetriever      │
                                     │  (Reciprocal Rank Fusion)   │
                                     └──────────────┬──────────────┘
                                                    │
                                                    ▼
                                     ┌─────────────────────────────┐
                                     │  Top-5 Chunks Relevantes    │
                                     └─────────────────────────────┘
```

### Componentes Clave del Módulo 4

| Componente | Archivo | Descripción |
|------------|---------|-------------|
| **Setup Cloud** | `setup_pinecone.py` | Crea y verifica el índice Serverless en Pinecone (AWS `us-east-1`) |
| **Infraestructura** | `infrastructure/pinecone_store.py` | Conexión con SDK de Pinecone, sincronización de dimensiones y LangChain VectorStore |
| **Ingesta Inteligente** | `ingest_pinecone.py` | Chunking por tokens, metadatos avanzados (`source`, `category`, `char_count`, `text`) y upsert en batches |
| **Recuperador Híbrido** | `core/rag_system.py` | Clase `RAGSystem` que unifica `BM25Retriever` y `PineconeVectorStore` con `EnsembleRetriever` |
| **Benchmark Golden Set** | `data/golden_set.json` | 5 casos de prueba canónicos con preguntas y fuentes esperadas |
| **Esquemas de Evaluación** | `domain/schemas/evaluation.py` | Validación Pydantic del benchmark y métricas de evaluación |
| **Script de Evaluación** | `evaluate.py` | Ejecución del benchmark, cálculo de Recall@5, Precision@5, Hit Rate, MRR y reporte en consola |

---

### Instrucciones para Replicar el Índice en Pinecone

Para replicar el índice y el flujo completo en la nube desde cero, sigue estos pasos:

#### 1. Configurar variables de entorno
Asegúrate de contar con tus claves de API en `app/.env`:
```env
PINECONE_API_KEY=tu_api_key_de_pinecone
PINECONE_INDEX_NAME=coderhouse-rag-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=dev-technical-docs
```

#### 2. Inicializar el índice en Pinecone Serverless
Ejecuta el script de setup:
```bash
PYTHONPATH=. python app/setup_pinecone.py
```
> El script comprueba automáticamente si el índice ya existe en tu cuenta de Pinecone. Si no existe, lo crea en modo Serverless (AWS `us-east-1`, métrica `cosine`) y espera hasta que el estado sea `ready`.

#### 3. Ejecutar el Pipeline de Ingesta Inteligente
Carga los documentos técnicos de `app/data/`, aplica fragmentación basada en tokens (~500 tokens), genera metadatos enriquecidos y realiza el upsert a Pinecone:
```bash
PYTHONPATH=. python app/ingest_pinecone.py
```
> **Nota de optimización:** Si el namespace ya contiene vectores, el script omite la re-ingesta para no duplicar datos ni incurrir en costos innecesarios. Para forzar una re-indexación limpia, añade la bandera `--force`:
> ```bash
> PYTHONPATH=. python app/ingest_pinecone.py --force
> ```

#### 4. Ejecutar la Evaluación Cuantitativa de Métricas
Evalúa el comportamiento del recuperador híbrido frente al Golden Set de preguntas de prueba:
```bash
PYTHONPATH=. python app/evaluate.py
```

---

### Prevención de Errores Comunes de Arquitectura

1. **Mismatch de Dimensiones:**
   - La arquitectura detecta automáticamente el modelo de embeddings activo y su dimensión (1536 para OpenAI `text-embedding-3-small`, 384 para HuggingFace local `all-MiniLM-L6-v2`). Si el índice en Pinecone tiene una dimensión distinta, el script lo detecta y resincroniza el índice para garantizar compatibilidad total.
2. **Uso Estricto de Namespaces:**
   - Todo el flujo de upsert y consulta utiliza el namespace configurable `dev-technical-docs`, asegurando segmentación multi-entorno y evitando ruido de datos.
3. **Chunking Equilibrado:**
   - Se utiliza `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con `chunk_size=500` tokens y `chunk_overlap=70`, garantizando que no se pierda el contexto ni se diluya la especificidad semántica ("Lost in the Middle").
4. **Almacenamiento de Texto en Metadatos:**
   - Cada vector incluye el contenido original dentro del campo `text` de sus metadatos en Pinecone, eliminando consultas secundarias a bases de datos relacionales durante el retrieval.

---

## Resultados de Evaluación del Módulo 4

A continuación se presentan los resultados obtenidos al ejecutar `app/evaluate.py` sobre el Golden Set de 5 casos canónicos técnicos:

### Tabla de Resultados por Caso de Prueba

| ID | Documento Esperado | Hit en Top-5? | Recall@5 | Precision@5 | Latencia Híbrida |
|:---|:-------------------|:-------------:|:--------:|:-----------:|:----------------:|
| **TC-001** (LCEL Pipe) | `langchain_intro.md` | ✅ SI | **1.00** | 0.40 | 911.2 ms |
| **TC-002** (Pydantic Ellipsis) | `pydantic_guide.md` | ✅ SI | **1.00** | 0.40 | 210.8 ms |
| **TC-003** (ChromaDB hnsw:space) | `chromadb_guide.md` | ✅ SI | **1.00** | 0.40 | 215.9 ms |
| **TC-004** (tiktoken & Chunking) | `rag_concepts.md` | ✅ SI | **1.00** | 0.40 | 213.7 ms |
| **TC-005** (Pinecone Serverless & BM25) | `pinecone_serverless_guide.md` | ✅ SI | **1.00** | 0.40 | 178.7 ms |

### Resumen Global del Benchmark

| Métrica | Valor Obtenido | Interpretación |
|:--------|:--------------:|:---------------|
| **Recall@5 Promedio** | **100.0%** | En el 100% de las consultas, el documento fuente canónico apareció dentro de los 5 fragmentos recuperados. |
| **Precision@5 Promedio** | **40.0%** | Proporción exacta de fragmentos relevantes sobre el total de 5 devueltos (dado el tamaño de los documentos, cada archivo aporta entre 2 y 3 fragmentos en total). |
| **Tasa de Acierto (Hit Rate@5)** | **100.0%** | Todas las preguntas del benchmark tuvieron al menos un acierto relevante. |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | En **todos los casos**, el fragmento más relevante del documento canónico quedó clasificado en la **posición #1** del ranking final de `EnsembleRetriever`. |
| **Latencia Promedio** | **346.05 ms** | Tiempo medio de respuesta incluyendo la llamada en red a Pinecone Serverless en AWS `us-east-1` y la búsqueda léxica BM25 combinadas. |

> El reporte detallado estructurado en JSON se genera automáticamente en `app/logs/evaluation_report.json`.
