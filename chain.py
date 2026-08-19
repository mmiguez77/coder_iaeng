"""
Módulo chain.py (app) - Re-exporta las funciones clave de la cadena LCEL del Módulo 2.
"""
from app.core.chain import (
    get_langchain_model,
    build_extraction_chain,
    process_text,
)

__all__ = ["get_langchain_model", "build_extraction_chain", "process_text"]
