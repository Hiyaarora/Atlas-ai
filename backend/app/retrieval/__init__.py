"""Retrieval strategies and the pipeline that combines them.

Imported by `services/retrieval_service.py`, which keeps its "question in,
ranked passages out" signature. Nothing in generation, citations or
conversation isolation depends on what happens in here, which is what makes
each retrieval change a drop-in.
"""

from app.retrieval.base import RetrievedChunk, Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import FusedChunk, reciprocal_rank_fusion
from app.retrieval.lexical import BM25Index, LexicalRetriever, tokenize
from app.retrieval.pipeline import hybrid_search

__all__ = [
    "BM25Index",
    "DenseRetriever",
    "FusedChunk",
    "LexicalRetriever",
    "RetrievedChunk",
    "Retriever",
    "hybrid_search",
    "reciprocal_rank_fusion",
    "tokenize",
]
