# Conceptos de RAG (Retrieval-Augmented Generation)

## ¿Qué es RAG?

RAG (Retrieval-Augmented Generation) es una arquitectura que combina la recuperación de información desde una base de conocimiento externa con la capacidad generativa de un modelo de lenguaje. En lugar de depender exclusivamente de la memoria paramétrica del LLM, RAG inyecta contexto relevante y verificable en cada consulta, reduciendo significativamente las alucinaciones.

## Flujo End-to-End de un sistema RAG

Un pipeline RAG típico sigue estos pasos:

1. **Ingesta de documentos**: Los documentos fuente (archivos de texto, PDFs, páginas web) se cargan y preparan para el procesamiento.
2. **Chunking (fragmentación)**: Los documentos se dividen en fragmentos más pequeños para optimizar la búsqueda y respetar los límites de contexto del LLM.
3. **Generación de embeddings**: Cada fragmento se convierte en un vector numérico (embedding) que captura su significado semántico.
4. **Indexación en vector store**: Los embeddings se almacenan en una base de datos vectorial para búsquedas eficientes por similitud.
5. **Recuperación**: Ante una consulta del usuario, se genera el embedding de la pregunta y se buscan los fragmentos más similares.
6. **Generación grounded**: Los fragmentos recuperados se inyectan como contexto en el prompt del LLM, que genera una respuesta fundamentada exclusivamente en esa información.

## Chunking: estrategias y consideraciones

El chunking es una etapa crítica que impacta directamente en la calidad de la recuperación. La herramienta más utilizada en LangChain es `RecursiveCharacterTextSplitter`, que intenta dividir el texto respetando la estructura natural del documento (párrafos, oraciones, palabras).

Parámetros clave del chunking:

- **chunk_size**: El tamaño máximo de cada fragmento. Se recomienda medirlo en tokens (no en caracteres) usando `from_tiktoken_encoder`, ya que el límite de contexto del LLM se mide en tokens.
- **chunk_overlap**: La cantidad de tokens compartidos entre fragmentos consecutivos. Un overlap de 50-100 tokens ayuda a preservar el contexto en los bordes de los fragmentos.

Un error común es usar chunks demasiado grandes (más de 1000 tokens), lo que diluye la relevancia de la información recuperada y puede causar el problema conocido como "Lost in the Middle", donde el LLM pierde atención en los fragmentos intermedios de un contexto largo.

## Embeddings

Los embeddings son representaciones vectoriales densas del texto que capturan similitud semántica. Dos textos con significado similar tendrán vectores cercanos en el espacio dimensional, incluso si usan palabras diferentes.

Es fundamental usar el mismo modelo de embeddings tanto para indexar los documentos como para generar el embedding de la consulta del usuario. Mezclar modelos (por ejemplo, indexar con OpenAI y consultar con HuggingFace) produce distancias vectoriales sin sentido y resultados de búsqueda aleatorios.

Modelos de embeddings populares incluyen `text-embedding-3-small` de OpenAI y `sentence-transformers/all-MiniLM-L6-v2` de HuggingFace, siendo este último gratuito y ejecutable localmente.
