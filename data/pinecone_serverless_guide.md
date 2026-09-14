# Guía de Pinecone Serverless y Búsqueda Híbrida

## ¿Qué es Pinecone Serverless?

Pinecone Serverless es una base de datos vectorial administrada y nativa en la nube, diseñada para escalar el almacenamiento y cómputo de embeddings de forma desacoplada y elástica. Permite indexar millones de vectores con latencias de milisegundos sin la necesidad de aprovisionar ni gestionar instancias dedicadas o memoria RAM persistente.

## Separación de Almacenamiento y Cómputo

A diferencia de las arquitecturas de bases de datos vectoriales tradicionales o locales, Pinecone Serverless separa completamente la capa de persistencia (que utiliza almacenamiento de objetos en la nube de alta durabilidad como AWS S3) de los nodos de indexación y consulta bajo demanda. Esto permite:
1. Pagar exclusivamente por los recursos consumidos y el almacenamiento real.
2. Escalar horizontalmente la tasa de consultas por segundo (QPS) de manera instantánea.
3. Garantizar persistencia duradera sin riesgo de degradación de hardware local.

## Namespaces y Multi-Tenancy

Los Namespaces (espacios de nombres) son particiones lógicas dentro de un mismo índice de Pinecone. Resultan críticos para:
- **Multi-tenancy**: Aislar la información de distintos clientes o empresas dentro del mismo índice sin mezclar datos.
- **Entornos**: Separar datos de desarrollo (`dev`), staging y producción (`prod`).
- **Rendimiento**: Evitar consultas ruidosas limitando el espacio de búsqueda vectorial al segmento de interés.

## Metadatos Enriquecidos e Ingesta Inteligente

En lugar de almacenar únicamente el vector de embeddings y un ID, Pinecone permite adjuntar un payload JSON con metadatos a cada registro. Las mejores prácticas recomiendan incluir el texto original (`text`), la fuente (`source`), la categoría y la fecha de ingesta dentro de los metadatos. Esto elimina la necesidad de realizar consultas complementarias a una base de datos relacional externa durante la etapa de retrieval.

## Búsqueda Híbrida (Hybrid Retrieval): Dense + Sparse

La búsqueda puramente semántica (vectores densos) destaca por capturar similitud conceptual y sinónimos, pero puede fallar ante identificadores exactos, números de serie, nombres de funciones o acrónimos técnicos específicos. Por otro lado, la búsqueda léxica basada en BM25 sobresale en coincidencias exactas por frecuencia de términos inversa (TF-IDF ponderado).

El **Recuperador Híbrido** (`EnsembleRetriever`) combina ambos paradigmas:
- **BM25Retriever**: Recupera los fragmentos con mayor densidad de palabras clave exactas.
- **VectorRetriever**: Recupera los fragmentos con mayor cercanía en el espacio semántico latente.
- **Reciprocal Rank Fusion (RRF)**: Fusiona las listas de resultados ponderando sus posiciones relativas mediante pesos configurables para emitir los Top-k fragmentos definitivos.
