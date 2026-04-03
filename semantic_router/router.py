"""
semantic_router/router.py — Intent Classification v6
======================================================
Phân loại intent câu hỏi bằng cosine similarity với sample embeddings.
Không cần LLM, không cần Qdrant — nhanh (<5ms sau khi build).

4 routes:
  job_search      : tìm việc + hỏi lương (pipeline RAG)
  career_advice   : tư vấn lộ trình, kỹ năng, định hướng
  career_guidance : intake flow 3 bước → phân tích cá nhân hóa
  chitchat        : chào hỏi, hội thoại thông thường

Cách hoạt động:
  1. Build: embed tất cả samples → ma trận embedding mỗi route
  2. Inference: embed query → cosine sim với từng route → chọn max
  3. Fallback: score < threshold → chitchat
"""

import os
import json

import numpy as np
from dataclasses import dataclass

from embedding_model.core import EmbeddingModel


@dataclass
class Route:
    name:    str
    samples: list[str]


_INTENT_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "intents.json")
)

ROUTES: list[Route] = []
try:
    with open(_INTENT_FILE, "r", encoding="utf-8") as f:
        _data = json.load(f)
    for name, samples in _data.items():
        ROUTES.append(Route(name=name, samples=samples))
    print(f"[Router] Loaded {len(ROUTES)} routes from intents.json")
except Exception as e:
    print(f"[Router] ⚠️ Lỗi đọc {_INTENT_FILE}: {e}")
    # Fallback minimal routes
    ROUTES = [
        Route("job_search",   ["tìm việc", "việc làm", "tuyển dụng"]),
        Route("career_advice",["tư vấn", "lộ trình", "kỹ năng cần học"]),
        Route("chitchat",     ["xin chào", "hello", "cảm ơn"]),
    ]


class SemanticRouter:
    """
    Phân loại intent bằng embedding similarity — không cần LLM.

    Dùng:
        router = SemanticRouter(embedding_model)
        route, confidence = router.guide("tìm việc python hà nội")
        # → ("job_search", 0.82)
    """

    THRESHOLD = 0.35  # dưới ngưỡng này → chitchat

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model   = embedding_model
        self._route_embeddings: dict[str, np.ndarray] = {}
        self._build()

    def _build(self):
        """Embed tất cả samples và normalize — chạy 1 lần khi khởi động."""
        print("[Router] Building route embeddings...", end=" ", flush=True)
        for route in ROUTES:
            embs  = self.embedding_model.get_query_embeddings_batch(route.samples)
            mat   = np.array(embs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._route_embeddings[route.name] = mat / norms
        print(f"✅ | {len(ROUTES)} routes")

    def guide(self, query: str) -> tuple[str, float]:
        """
        Phân loại câu hỏi → (route_name, confidence_score).

        Dùng max similarity (không phải mean) — ổn định hơn với query ngắn.
        Nếu score < THRESHOLD → trả về chitchat.
        """
        if not query or not query.strip():
            return "chitchat", 0.0

        q_vec  = np.array(
            self.embedding_model.get_query_embedding(query), dtype=np.float32
        )
        norm   = np.linalg.norm(q_vec)
        if norm == 0:
            return "chitchat", 0.0
        q_vec  = q_vec / norm

        best_route = "chitchat"
        best_score = 0.0

        for route in ROUTES:
            mat   = self._route_embeddings.get(route.name)
            if mat is None:
                continue
            sim   = float(np.max(mat @ q_vec))
            if sim > best_score:
                best_score = sim
                best_route = route.name

        if best_score < self.THRESHOLD:
            return "chitchat", best_score

        return best_route, best_score
