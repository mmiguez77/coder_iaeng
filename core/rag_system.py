"""
Sistema RAG Escalable con Recuperación Híbrida (BM25 + Pinecone) — Módulo 4
============================================================================
Implementa la clase RAGSystem que encapsula un EnsembleRetriever de LangChain,
combinando búsqueda vectorial densa en Pinecone Serverless y búsqueda léxica
esparsa mediante BM25Retriever.
"""
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Asegurar que los módulos de 'app' sean importables independientemente del CWD
app_dir = Path(__file__).resolve().parent.parent
parent_dir = app_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from loguru import logger
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# Import dinámico y resiliente de EnsembleRetriever según versión de LangChain
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers.ensemble import EnsembleRetriever

from app.config.settings import get_settings
from app.infrastructure.pinecone_store import get_pinecone_vector_store
from app.ingest_pinecone import load_raw_documents, split_and_enrich_documents
from app.domain.schemas.rag_response import RAGResponse


class RAGSystem:
    """
    Sistema RAG empresarial que encapsula un EnsembleRetriever híbrido.
    Combina:
      1. Búsqueda léxica (BM25Retriever) para coincidencias exactas de términos técnicos y código.
      2. Búsqueda semántica (PineconeVectorStore) para similitud conceptual en la nube.
      3. Reciprocal Rank Fusion (RRF) para fusionar y clasificar los Top-k documentos.
    """

    def __init__(
        self,
        top_k: int = 5,
        weights: Optional[List[float]] = None,
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
    ):
        """
        Inicializa el sistema RAG híbrido.

        Args:
            top_k: Cantidad de documentos a recuperar (default 5).
            weights: Ponderación entre [BM25, Pinecone] (default [0.5, 0.5]).
            namespace: Namespace de Pinecone a consultar (default desde settings).
            index_name: Nombre del índice de Pinecone (default desde settings).
        """
        self.top_k = top_k
        self.weights = weights or [0.5, 0.5]
        self.settings = get_settings()
        self.namespace = namespace or self.settings.pinecone_namespace
        self.index_name = index_name or self.settings.pinecone_index_name

        logger.info(
            f"Inicializando RAGSystem (top_k={self.top_k}, pesos={self.weights}, "
            f"namespace='{self.namespace}', index='{self.index_name}')..."
        )

        # 1. Configurar recuperador léxico BM25
        logger.debug("Construyendo corpus de documentos para BM25Retriever...")
        raw_docs = load_raw_documents()
        self.chunks = split_and_enrich_documents(raw_docs)

        self.bm25_retriever = BM25Retriever.from_documents(
            documents=self.chunks,
            k=self.top_k,
        )
        logger.debug(f"BM25Retriever indexó {len(self.chunks)} fragmentos.")

        # 2. Configurar recuperador vectorial Pinecone
        logger.debug("Conectando con PineconeVectorStore...")
        self.vector_store = get_pinecone_vector_store(
            index_name=self.index_name,
            namespace=self.namespace,
        )
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": self.top_k,
                "namespace": self.namespace,
            }
        )
        logger.debug("PineconeVectorStore retriever configurado.")

        # 3. Ensamblar EnsembleRetriever
        logger.debug("Configurando EnsembleRetriever híbrido...")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=self.weights,
        )
        logger.info("RAGSystem inicializado exitosamente.")

    def retrieve(self, query: str) -> List[Document]:
        """
        Ejecuta la recuperación híbrida para una consulta y retorna los top-k documentos.

        Args:
            query: Texto de la consulta o pregunta técnica.

        Returns:
            Lista de objetos Document ordenados por relevancia tras Reciprocal Rank Fusion.
        """
        start_time = time.perf_counter()
        logger.debug(f"Ejecutando búsqueda híbrida para: \"{query}\"")

        # Invocación del EnsembleRetriever
        try:
            docs = self.ensemble_retriever.invoke(query)
        except Exception:
            # Compatibilidad con métodos anteriores
            docs = self.ensemble_retriever.get_relevant_documents(query)

        # Truncar a top_k exacto
        top_docs = docs[: self.top_k]
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            f"Recuperación híbrida completada en {elapsed_ms:.2f}ms. "
            f"Fragmentos recuperados: {len(top_docs)}"
        )
        return top_docs

    def retrieve_with_details(self, query: str) -> Dict[str, Any]:
        """
        Recupera los top-k documentos e incluye métricas detalladas de trazabilidad.
        """
        start_time = time.perf_counter()
        docs = self.retrieve(query)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        sources = [doc.metadata.get("source", "desconocido") for doc in docs]
        categories = [doc.metadata.get("category", "desconocido") for doc in docs]

        return {
            "query": query,
            "top_k": len(docs),
            "latencia_ms": elapsed_ms,
            "fuentes": sources,
            "categorias": categories,
            "documentos": docs,
        }


if __name__ == "__main__":
    # Permite probar consultas manuales directamente desde la terminal
    user_query = sys.argv[1] if len(sys.argv) > 1 else "¿Cómo se compone una cadena declarativa en LangChain Expression Language (LCEL)?"
    print(f"\nProbando consulta manual: '{user_query}'\n")
    system = RAGSystem(top_k=5)
    details = system.retrieve_with_details(user_query)

    print("=" * 70)
    print(f"RESULTADOS DE RECUPERACIÓN HÍBRIDA (Top-{details['top_k']})")
    print(f"Latencia: {details['latencia_ms']:.2f} ms")
    print("=" * 70)
    for i, doc in enumerate(details["documentos"], 1):
        src = doc.metadata.get("source", "desconocido")
        cat = doc.metadata.get("category", "desconocido")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        snippet = doc.page_content.strip().replace("\n", " ")[:150]
        print(f"\n[{i}] Fuente: {src} | Categoría: {cat} | Chunk #{chunk_idx}")
        print(f"    Texto: {snippet}...")
    print("\n" + "=" * 70 + "\n")
