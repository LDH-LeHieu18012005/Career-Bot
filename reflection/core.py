"""
reflection/core.py — Chat History + Query Rewriting v6
========================================================
Hai nhiệm vụ:
  1. Lưu lịch sử hội thoại vào Qdrant (không cần vector — dùng collection riêng)
  2. Viết lại query dựa trên ngữ cảnh hội thoại → standalone query đầy đủ nghĩa

Tại sao dùng Qdrant thay vì RAM?
  - Không cần cài thêm service
  - Session persist qua restart
  - Đủ nhanh cho session-based read/write

Fixes từ v4:
  [FIX-1] Rolling window đúng: scroll TẤT CẢ messages → sort → slice N turns cuối
  [FIX-2] clear_session() dùng FilterSelector đúng cách
  [FIX-3] Trim session sau mỗi save_turn() để giữ rolling window
"""

import os
import uuid
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    FilterSelector,
)

load_dotenv()

QDRANT_URL      = os.getenv("QDRANT_URL", "")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")
HISTORY_COLLECTION = "career_chat_history"

_ROLE_MAP = {"human": "user", "ai": "assistant"}

_MAX_HISTORY_TURNS = 10   # giữ tối đa N turns (= 2N messages) mỗi session


class SelfReflection:
    """
    Quản lý lịch sử hội thoại và query rewriting.

    Dùng:
        reflection = SelfReflection(llm=llm_client)
        reflection.save_turn(session_id, "câu hỏi", "human")
        reflection.save_turn(session_id, "câu trả lời", "ai")
        standalone = reflection.process_query(session_id, "câu hỏi mới")
        history    = reflection.get_history(session_id)
    """

    def __init__(self, llm, history_collection: str = HISTORY_COLLECTION):
        self.llm      = llm
        self.hist_col = history_collection
        self.client   = self._connect()
        self._ready   = False
        self._ensure_collection()

    # ── Kết nối ───────────────────────────────────────────────────────────────

    def _connect(self) -> QdrantClient:
        if QDRANT_URL:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=15)
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=15)

    def _ensure_collection(self) -> bool:
        """Tạo history collection nếu chưa có. Retry 3 lần nếu lỗi kết nối."""
        for attempt in range(3):
            try:
                existing = [c.name for c in self.client.get_collections().collections]
                if self.hist_col not in existing:
                    self.client.create_collection(
                        collection_name=self.hist_col,
                        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
                    )
                    print(f"[Reflection] Created collection '{self.hist_col}'")
                self._ready = True
                return True
            except Exception as e:
                print(f"[Reflection] Attempt {attempt+1}/3 — {e}")
                time.sleep(2 ** attempt)
        print("[Reflection] ⚠️ Không kết nối được Qdrant — history bị tắt")
        return False

    # ── Ghi lịch sử ───────────────────────────────────────────────────────────

    def save_turn(self, session_id: str, content: str, role: str):
        """
        Lưu 1 turn (human hoặc ai) vào history collection.
        Role: "human" hoặc "ai"
        """
        if not self._ready or not content:
            return
        try:
            point = PointStruct(
                id     = str(uuid.uuid4()),
                vector = [0.0],  # dummy vector
                payload={
                    "session_id": session_id,
                    "role":       _ROLE_MAP.get(role, role),
                    "content":    content[:2000],  # truncate dài quá
                    "created_at": time.time(),
                },
            )
            self.client.upsert(collection_name=self.hist_col, points=[point])
            self._trim_session(session_id)
        except Exception as e:
            print(f"[Reflection] save_turn error: {e}")

    def _trim_session(self, session_id: str):
        """
        [FIX-3] Giữ rolling window: xóa messages cũ nếu vượt quá giới hạn.
        """
        try:
            all_msgs = self._scroll_session(session_id)
            max_msgs = _MAX_HISTORY_TURNS * 2  # N turns = 2N messages

            if len(all_msgs) <= max_msgs:
                return

            # Xóa những message cũ nhất
            to_delete = [m["id"] for m in all_msgs[:-max_msgs]]
            if to_delete:
                self.client.delete(
                    collection_name=self.hist_col,
                    points_selector=to_delete,
                )
        except Exception as e:
            print(f"[Reflection] trim error: {e}")

    # ── Đọc lịch sử ───────────────────────────────────────────────────────────

    def _scroll_session(self, session_id: str) -> list[dict]:
        """
        [FIX-1] Scroll TẤT CẢ messages của session, sort theo created_at.
        """
        if not self._ready:
            return []

        msgs = []
        offset = None
        while True:
            try:
                batch, offset = self.client.scroll(
                    collection_name=self.hist_col,
                    scroll_filter=Filter(must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id),
                        )
                    ]),
                    limit=200,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                print(f"[Reflection] scroll error: {e}")
                break

            for point in batch:
                msgs.append({
                    "id":         point.id,
                    "role":       point.payload.get("role", "user"),
                    "content":    point.payload.get("content", ""),
                    "created_at": point.payload.get("created_at", 0),
                })

            if offset is None:
                break

        return sorted(msgs, key=lambda m: m["created_at"])

    def get_history(self, session_id: str) -> list[dict]:
        """
        Trả về history dạng OpenAI messages format:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        msgs = self._scroll_session(session_id)
        # Lấy N turns cuối
        msgs = msgs[-(_MAX_HISTORY_TURNS * 2):]
        return [{"role": m["role"], "content": m["content"]} for m in msgs]

    # ── Query Rewriting ───────────────────────────────────────────────────────

    def process_query(self, session_id: str, query: str) -> str:
        """
        Viết lại query dựa trên lịch sử → standalone query đầy đủ nghĩa.

        Ví dụ:
          History:  "tìm việc python hà nội"
          Query:    "còn cái nào lương cao hơn không"
          Output:   "tìm việc python hà nội lương cao"
        """
        if not self._ready:
            return query

        history = self.get_history(session_id)
        if not history:
            return query

        # Chỉ lấy 4 turns gần nhất để rewrite
        recent = history[-4:]
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in recent
        )

        prompt = (
            f"Lịch sử hội thoại:\n{history_text}\n\n"
            f"Câu hỏi mới: {query}\n\n"
            "Viết lại câu hỏi trên thành 1 câu tìm kiếm độc lập, đầy đủ nghĩa, "
            "không cần ngữ cảnh. Chỉ trả lời câu đó, không giải thích."
        )

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
            )
            rewritten = resp.choices[0].message.content.strip()
            # Sanity check: không trả về quá dài hoặc quá khác
            if 5 <= len(rewritten) <= 200:
                print(f"[Reflection] Rewrite: '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            print(f"[Reflection] rewrite error: {e}")

        return query

    # ── Session management ────────────────────────────────────────────────────

    def clear_session(self, session_id: str):
        """[FIX-2] Xóa toàn bộ history của 1 session."""
        if not self._ready:
            return
        try:
            self.client.delete(
                collection_name=self.hist_col,
                points_selector=FilterSelector(
                    filter=Filter(must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id),
                        )
                    ])
                ),
            )
            print(f"[Reflection] Cleared session '{session_id}'")
        except Exception as e:
            print(f"[Reflection] clear_session error: {e}")
