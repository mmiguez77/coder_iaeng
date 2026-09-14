"""
Script de Evaluación Cuantitativa del Sistema RAG — Módulo 4
============================================================
Evalúa el recuperador híbrido (BM25 + Pinecone Serverless) utilizando un
Golden Set de prueba (app/data/golden_set.json) y calcula las métricas
fundamentales de recuperación:
  - Recall@5: ¿Está el documento correcto entre los 5 recuperados?
  - Precision@5: Proporción de los 5 fragmentos recuperados que son relevantes.
  - Hit Rate@5: Tasa de acierto de recuperación en los Top-5.
  - MRR (Mean Reciprocal Rank): Posición del primer documento relevante.

Ejecución desde la raíz del repositorio:
    PYTHONPATH=. python app/evaluate.py
"""
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Asegurar que los módulos de 'app' sean importables independientemente del CWD
app_dir = Path(__file__).resolve().parent
parent_dir = app_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from loguru import logger
from app.core.rag_system import RAGSystem
from app.domain.schemas.evaluation import (
    GoldenSet,
    RetrievalEvaluationResult,
    BenchmarkSummary,
)

# --------------------------------------------------------------------------- #
#  Configuración de logs
# --------------------------------------------------------------------------- #
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "execution.log"

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:DD/MM/YY - HH.mm.ss}</green> | <level>{level: <8}</level> | {function}:{line} - {message}",
    level="INFO",
)
logger.add(
    str(log_file_path),
    format="{time:DD/MM/YY - HH.mm.ss} | {level: <8} | {function}:{line} - {message}",
    level="INFO",
    rotation="10 MB",
)

GOLDEN_SET_PATH = Path(__file__).parent / "data" / "golden_set.json"


def load_golden_set(file_path: Path = GOLDEN_SET_PATH) -> GoldenSet:
    """Carga y valida el archivo del Golden Set mediante Pydantic."""
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo del Golden Set en: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GoldenSet.model_validate(data)


def evaluate_retrieval(top_k: int = 5) -> BenchmarkSummary:
    """
    Ejecuta el benchmark completo calculando Recall@k y Precision@k sobre el Golden Set.
    """
    logger.info("=== INICIANDO EVALUACIÓN CUANTITATIVA DEL SISTEMA RAG (MÓDULO 4) ===")
    golden_set = load_golden_set()
    logger.info(f"Golden Set cargado: {len(golden_set.casos)} casos de prueba.")

    # Inicializar el sistema RAG híbrido
    rag_system = RAGSystem(top_k=top_k)

    eval_results: List[RetrievalEvaluationResult] = []
    reciprocal_ranks: List[float] = []

    for idx, test_case in enumerate(golden_set.casos, 1):
        logger.info(f"[{idx}/{len(golden_set.casos)}] Evaluando pregunta: '{test_case.pregunta}'")
        start_t = time.perf_counter()

        # Recuperar Top-K documentos
        retrieved_docs = rag_system.retrieve(test_case.pregunta)
        latency_ms = (time.perf_counter() - start_t) * 1000

        # Fuentes recuperadas
        retrieved_sources = [
            doc.metadata.get("source", "desconocido") for doc in retrieved_docs
        ]

        # 1. Recall@k: ¿Aparece el documento esperado en el Top-k?
        hit = test_case.documento_esperado in retrieved_sources
        recall_at_k = 1.0 if hit else 0.0

        # 2. Precision@k: Proporción de los k documentos recuperados que corresponden al doc esperado
        relevant_count = sum(1 for src in retrieved_sources if src == test_case.documento_esperado)
        precision_at_k = relevant_count / top_k if top_k > 0 else 0.0

        # 3. Reciprocal Rank (MRR component)
        if hit:
            first_rank = retrieved_sources.index(test_case.documento_esperado) + 1
            rr = 1.0 / first_rank
        else:
            rr = 0.0
        reciprocal_ranks.append(rr)

        result_item = RetrievalEvaluationResult(
            id=test_case.id,
            pregunta=test_case.pregunta,
            documento_esperado=test_case.documento_esperado,
            documentos_recuperados=retrieved_sources,
            hit=hit,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            latencia_ms=latency_ms,
        )
        eval_results.append(result_item)

    # Métricas agregadas
    total = len(eval_results)
    mean_recall = sum(r.recall_at_k for r in eval_results) / total if total > 0 else 0.0
    mean_precision = sum(r.precision_at_k for r in eval_results) / total if total > 0 else 0.0
    hit_rate = sum(1 for r in eval_results if r.hit) / total if total > 0 else 0.0
    avg_latency = sum(r.latencia_ms for r in eval_results) / total if total > 0 else 0.0
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0

    summary = BenchmarkSummary(
        total_consultas=total,
        k=top_k,
        recall_promedio=round(mean_recall, 4),
        precision_promedio=round(mean_precision, 4),
        tasa_exito_hit=round(hit_rate, 4),
        latencia_promedio_ms=round(avg_latency, 2),
        resultados=eval_results,
    )

    print_report(summary, mrr=mrr)
    return summary


def print_report(summary: BenchmarkSummary, mrr: float = 0.0):
    """
    Imprime un informe tabular formateado en consola con el desglose de métricas.
    """
    print("\n" + "=" * 80)
    print(f"        REPORTE DE EVALUACIÓN CUANTITATIVA — SISTEMA RAG HÍBRIDO (TOP-{summary.k})")
    print("=" * 80)

    # Encabezado de tabla
    header = f"{'ID':<8} | {'Documento Esperado':<22} | {'Hit?':<6} | {'Recall@5':<10} | {'Prec@5':<8} | {'Latencia':<10}"
    print(header)
    print("-" * 80)

    for r in summary.resultados:
        hit_icon = "✅ SI" if r.hit else "❌ NO"
        row = (
            f"{r.id:<8} | "
            f"{r.documento_esperado:<22} | "
            f"{hit_icon:<6} | "
            f"{r.recall_at_k:<10.2f} | "
            f"{r.precision_at_k:<8.2f} | "
            f"{r.latencia_ms:>7.1f} ms"
        )
        print(row)

    print("-" * 80)
    print(f"RESUMEN GLOBAL DEL BENCHMARK ({summary.total_consultas} CONSULTAS):")
    print(f"  • Recall@{summary.k} Promedio:    {summary.recall_promedio * 100:.1f}%")
    print(f"  • Precision@{summary.k} Promedio: {summary.precision_promedio * 100:.1f}%")
    print(f"  • Tasa de Acierto (Hit Rate):  {summary.tasa_exito_hit * 100:.1f}%")
    print(f"  • Mean Reciprocal Rank (MRR):  {mrr:.4f}")
    print(f"  • Latencia Promedio:          {summary.latencia_promedio_ms:.2f} ms")
    print("=" * 80 + "\n")

    # Guardar reporte JSON en logs para trazabilidad
    report_file = Path(__file__).parent / "logs" / "evaluation_report.json"
    report_data = summary.model_dump()
    report_data["mrr"] = round(mrr, 4)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Reporte detallado guardado en: {report_file}")


if __name__ == "__main__":
    evaluate_retrieval(top_k=5)
