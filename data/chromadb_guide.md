# Guía de ChromaDB

## ¿Qué es ChromaDB?

ChromaDB es una base de datos vectorial de código abierto diseñada para almacenar y consultar embeddings de forma eficiente. Es especialmente popular en aplicaciones de RAG porque ofrece persistencia local sin necesidad de infraestructura externa, instalación simple como paquete de Python, y una API intuitiva compatible con LangChain.

## Cliente persistente

ChromaDB soporta dos modos de operación: en memoria (efímero) y persistente (datos guardados en disco). Para aplicaciones RAG, el modo persistente es esencial ya que permite reutilizar los embeddings indexados entre ejecuciones sin volver a procesar los documentos.

```python
import chromadb

client = chromadb.PersistentClient(path="./vectorstore")
```

El directorio especificado en `path` contendrá los archivos de la base de datos SQLite y los índices vectoriales. Esta carpeta debe incluirse en `.gitignore` ya que los datos se generan en runtime.

## Colecciones

Una colección en ChromaDB es el equivalente a una tabla en bases de datos relacionales. Agrupa documentos, sus embeddings y metadatos asociados bajo un nombre único.

```python
collection = client.get_or_create_collection(
    name="mi_coleccion",
    metadata={"hnsw:space": "cosine"}
)
```

El parámetro `hnsw:space` define la métrica de distancia para la búsqueda. Las opciones son `cosine` (similitud coseno, la más común), `l2` (distancia euclidiana) e `ip` (producto interno).

## Operaciones principales

### Agregar documentos
```python
collection.add(
    documents=["texto del fragmento 1", "texto del fragmento 2"],
    metadatas=[{"source": "archivo1.md"}, {"source": "archivo2.md"}],
    ids=["id1", "id2"]
)
```

Cada documento requiere un ID único. Los metadatos permiten almacenar información adicional como el archivo de origen, la fecha de indexación o la posición del fragmento en el documento original.

### Consultar por similitud
```python
results = collection.query(
    query_texts=["mi pregunta"],
    n_results=4
)
```

La consulta convierte el texto a un embedding usando el mismo modelo utilizado durante la indexación y retorna los `n_results` fragmentos más cercanos semánticamente.

## Integración con LangChain

LangChain proporciona el wrapper `Chroma` (del paquete `langchain-chroma`) que simplifica la interacción con ChromaDB y expone un método `.as_retriever()` para convertir el vector store en un componente compatible con cadenas LCEL.

```python
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="mi_coleccion",
    embedding_function=embeddings,
    persist_directory="./vectorstore"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

El parámetro `k` controla cuántos fragmentos se recuperan por consulta. Se recomienda mantenerlo entre 3 y 5 para evitar el problema del "contexto infinito", donde demasiados fragmentos degradan la atención del modelo y aumentan los costos de tokens.

## Verificación de índice existente

Una buena práctica es verificar si la colección ya fue indexada antes de volver a procesar los documentos, optimizando tiempo y recursos:

```python
if collection.count() > 0:
    print("Índice existente — omitiendo re-indexación")
else:
    # Indexar documentos por primera vez
```
