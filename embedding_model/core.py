"""
embedding_model/core.py — Jina v3 Wrapper (Phase 2 — Query Embedding)
=======================================================================
Chỉ dùng để embed QUERY (1 câu/lần chat, CPU đủ nhanh ~0.3-0.5s).
Embed documents (ingest) được thực hiện ở Phase 1 trên Kaggle.

v6: Hỗ trợ batch query embedding cho SemanticRouter._build()
"""

import os
import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()

HF_MODEL_NAME = os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v3")
EMBED_DEVICE   = os.getenv("EMBED_DEVICE", "")
EMBED_DIM      = 1024  # cố định theo jina-v3


def _resolve_device() -> str:
    if EMBED_DEVICE:
        return EMBED_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingModel:
    """
    Wrapper Jina v3 dùng sentence-transformers.

    Dùng:
        model = EmbeddingModel()
        vec   = model.get_query_embedding("tìm việc python hà nội")
        vecs  = model.get_query_embeddings_batch(["query1", "query2"])
    """

    def __init__(self):
        self._device    = _resolve_device()
        self._model     = self._load()
        self.dimension  = EMBED_DIM

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Cài sentence-transformers:\n"
                "pip install sentence-transformers"
            )

        print(f"[Embedder] Loading {HF_MODEL_NAME} | device={self._device} ...")
        try:
            model = SentenceTransformer(
                HF_MODEL_NAME,
                trust_remote_code=True,
                device=self._device,
            )
            print("[Embedder] ✅ Model loaded")
            return model
        except Exception as e:
            raise RuntimeError(f"[Embedder] Không load được model: {e}")

    def get_query_embedding(self, text: str) -> list[float]:
        """Embed 1 query → list[float] (1024 dims)."""
        if not text or not text.strip():
            return [0.0] * EMBED_DIM
        try:
            vec = self._model.encode(
                text,
                normalize_embeddings=True,
                task="retrieval.query",
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vec.tolist()
        except Exception as e:
            print(f"[Embedder] Lỗi embed query: {e}")
            return [0.0] * EMBED_DIM

    def get_query_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed nhiều queries cùng lúc — dùng cho SemanticRouter._build()."""
        if not texts:
            return []
        try:
            vecs = self._model.encode(
                texts,
                normalize_embeddings=True,
                task="retrieval.query",
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return vecs.tolist()
        except Exception as e:
            print(f"[Embedder] Lỗi batch embed: {e}")
            return [[0.0] * EMBED_DIM for _ in texts]

    def close(self):
        """Giải phóng VRAM nếu dùng GPU."""
        try:
            del self._model
            if self._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
