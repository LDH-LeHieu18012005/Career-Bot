"""
flask_serve.py — Career Bot API Server v6.3
============================================
[V6.3] Cải tiến:
  - Prompt enforcement: filter mạnh (lương/KN/địa điểm), format cứng, chống bịa/lặp job
  - _build_rag_messages: truyền [BỘ LỌC USER YÊU CẦU] rõ ràng cho LLM
  - _ensure_format: xoá 5 pattern bẩn LLM phổ biến

[V6.2] Tối ưu tốc độ:
  - _HistoryCache: RAM cache (TTL 5 phút) → bỏ 1 Qdrant roundtrip/request
  - save_turn vẫn ghi Qdrant async

Entrypoint chính cho Phase 2 (chạy local).
"""

import os
import re
import sys
import threading
import traceback
import atexit
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

# ── Imports ───────────────────────────────────────────────────────────────────
from embedding_model.core      import EmbeddingModel
from hf_client                 import GroqClient
from rag.core                  import RAG
from reflection.core           import SelfReflection
from semantic_router.router    import SemanticRouter
from pipeline.deadline_cleaner import delete_expired
from prompts import (
    SYSTEM_JOB_RAG, SYSTEM_LINK_ONLY,
    SYSTEM_ADVICE_RAG, SYSTEM_CHAT,
)

app = Flask(__name__)
CORS(app)

# ── LLM config per route ──────────────────────────────────────────────────────

_ROUTE_LLM_CONFIG = {
    "job_search":    {"max_tokens": 1200, "temperature": 0.2},
    "link_only":     {"max_tokens": 600,  "temperature": 0.0},
    "career_advice": {"max_tokens": 1600, "temperature": 0.3},
    "chitchat":      {"max_tokens": 400,  "temperature": 0.7},
}
_DEFAULT_LLM_CONFIG = {"max_tokens": 800, "temperature": 0.5}

# ── Pipeline ──────────────────────────────────────────────────────────────────

_pipeline: dict | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> dict:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        print("[Pipeline] Initializing...")
        t0 = time.time()

        try:
            embedding  = EmbeddingModel()
            llm        = GroqClient()
            rag        = RAG(embedding_model=embedding)
            reflection = SelfReflection(llm=llm)
            router     = SemanticRouter(embedding)

            _pipeline = {
                "embedding":  embedding,
                "llm":        llm,
                "rag":        rag,
                "reflection": reflection,
                "router":     router,
            }

            print(f"[Pipeline] Ready in {time.time()-t0:.1f}s")
            return _pipeline

        except Exception as e:
            print(f"[Pipeline] ERROR: {e}")
            traceback.print_exc()
            if "huggingface" in str(e).lower() or "getaddrinfo" in str(e).lower():
                print("\n[WARNING] Không kết nối được HuggingFace. Kiểm tra:")
                print("   • Internet")
                print("   • USE_JINA_API=true và JINA_API_KEY trong .env\n")
            raise


# ── History RAM Cache (V6.2) ──────────────────────────────────────────────────

class _HistoryCache:
    _TTL   = 300
    _LIMIT = 20

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._cleanup, daemon=True, name="HistCache-GC").start()

    def get(self, sid: str, reflection) -> list[dict]:
        with self._lock:
            entry = self._store.get(sid)
        if entry and (time.time() - entry["ts"]) < self._TTL:
            return list(entry["messages"])

        msgs = reflection.get_history(sid)
        with self._lock:
            self._store[sid] = {"messages": list(msgs), "ts": time.time()}
        return msgs

    def push(self, sid: str, role: str, content: str):
        with self._lock:
            if sid not in self._store:
                self._store[sid] = {"messages": [], "ts": time.time()}
            msgs = self._store[sid]["messages"]
            msgs.append({"role": role, "content": content})
            if len(msgs) > self._LIMIT:
                self._store[sid]["messages"] = msgs[-self._LIMIT:]
            self._store[sid]["ts"] = time.time()

    def _cleanup(self):
        while True:
            time.sleep(600)
            now = time.time()
            with self._lock:
                expired = [k for k, v in self._store.items() if now - v["ts"] > self._TTL * 2]
                for k in expired:
                    del self._store[k]


_hist_cache = _HistoryCache()

# ── Session Context ───────────────────────────────────────────────────────────

_CTX_LOCK: threading.Lock = threading.Lock()
_CTX_STORE: dict[str, dict] = {}
_CTX_TTL = 7200


def _get_ctx(sid: str) -> dict:
    with _CTX_LOCK:
        return _CTX_STORE.get(sid, {})


def _save_ctx(sid: str, keyword: str, filters: dict, seen_ids: list | None = None, last_max_salary: float = 0.0):
    with _CTX_LOCK:
        prev = _CTX_STORE.get(sid, {})
        merged_ids = list(set(prev.get("seen_ids", []) + (seen_ids or [])))
        _CTX_STORE[sid] = {
            "keyword":        keyword,
            "filters":        filters,
            "seen_ids":       merged_ids[-30:],
            "last_max_salary": last_max_salary,
            "ts":             time.time(),
        }


def _cleanup_ctx():
    while True:
        time.sleep(1800)
        now = time.time()
        with _CTX_LOCK:
            expired = [k for k, v in _CTX_STORE.items() if now - v.get("ts", 0) > _CTX_TTL]
            for k in expired:
                del _CTX_STORE[k]
        if expired:
            print(f"[CTX] Cleaned {len(expired)} expired sessions")


threading.Thread(target=_cleanup_ctx, daemon=True).start()

# ── Query Parser ──────────────────────────────────────────────────────────────

_LOCATION_KEYWORDS = {
    "hà nội": "ha_noi", "hanoi": "ha_noi", "hn": "ha_noi",
    "hồ chí minh": "ho_chi_minh", "hcm": "ho_chi_minh",
    "tp.hcm": "ho_chi_minh", "sài gòn": "ho_chi_minh",
    "đà nẵng": "da_nang",
    "cần thơ": "can_tho", "hải phòng": "hai_phong",
    "bình dương": "binh_duong", "đồng nai": "dong_nai",
    "remote": "remote", "work from home": "remote", "wfh": "remote",
}

_SALARY_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:[-–đến]\s*(\d+(?:[.,]\d+)?))?\s*(triệu|tr|million|m)",
    re.IGNORECASE,
)

_LINK_ONLY_KW = ["link", "url", "ứng tuyển", "apply", "xin link"]
_FRESHER_KW   = ["fresher", "mới ra trường", "chưa có kinh nghiệm", "sinh viên", "intern",
                  "mới học", "tự học", "chưa biết gì", "học việc", "thực tập"]
_JUNIOR_KW    = ["junior", "1 năm", "2 năm", "mới đi làm"]
_SENIOR_KW    = ["senior", "lead", "3 năm", "4 năm", "5 năm", "nhiều năm"]


def _parse_query(query: str) -> tuple[str, dict]:
    q = query.lower()
    filters: dict = {}

    for kw, norm in _LOCATION_KEYWORDS.items():
        if kw in q:
            filters["location_norm"] = norm
            break

    m = _SALARY_RE.search(q)
    if m:
        lo = float(m.group(1).replace(",", "."))
        hi = float(m.group(2).replace(",", ".")) if m.group(2) else lo
        if lo > 500:
            lo /= 1_000_000
        if hi > 500:
            hi /= 1_000_000
        filters["salary_min"] = round(min(lo, hi), 1)

    if any(k in q for k in _FRESHER_KW):
        filters["experience_norm"] = "fresher"
    elif any(k in q for k in _JUNIOR_KW):
        filters["experience_norm"] = "junior"
    elif any(k in q for k in _SENIOR_KW):
        filters["experience_norm"] = "senior"

    if any(k in q for k in _LINK_ONLY_KW):
        filters["link_only"] = True

    keyword = query
    for kw in _LOCATION_KEYWORDS:
        keyword = keyword.replace(kw, "")
    keyword = _SALARY_RE.sub("", keyword)
    keyword = re.sub(r"\s+", " ", keyword).strip()

    return keyword, filters


def _merge_ctx(sid: str, query: str, new_kw: str, new_f: dict) -> tuple[str, dict]:
    ctx = _get_ctx(sid)
    if not ctx:
        return new_kw or query, new_f

    merged_f = dict(ctx.get("filters", {}))
    merged_f.update({k: v for k, v in new_f.items() if v is not None})

    if any(k in query.lower() for k in ["cao hơn", "tốt hơn", "lương cao", "nhiều hơn"]):
        last_max = ctx.get("last_max_salary", 0.0)
        if last_max > 0:
            merged_f["salary_min"] = round(last_max * 1.1, 1)

    merged_kw = new_kw if len(new_kw) >= 4 else ctx.get("keyword", new_kw or query)
    return merged_kw, merged_f


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/api/v1/health", methods=["GET"])
def health():
    p = get_pipeline()
    return jsonify({
        "status":     "ok",
        "rag_count":  p["rag"].collection_count(),
        "bm25_status": p["rag"].bm25_status(),
    })


@app.route("/api/v1/chat", methods=["POST"])
def chat():
    data  = request.get_json(force=True) or {}
    query = (data.get("query") or data.get("message") or "").strip()
    sid   = (data.get("session_id") or "default").strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        return _handle(query, sid)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Core Handler ──────────────────────────────────────────────────────────────

def _handle(query: str, sid: str):
    p = get_pipeline()

    route, conf = p["router"].guide(query)
    print(f"[Handler] route={route} | conf={conf:.2f}")

    history = _hist_cache.get(sid, p["reflection"])
    standalone = _smart_rewrite(query, sid, route, p, history)

    valid_links = set()
    filters = {}

    if route in ("job_search", "career_advice"):
        ctx = _get_ctx(sid)
        exclude_ids = ctx.get("seen_ids", [])
        rag_msgs, valid_links, current_ids, max_sal, filters = _build_rag_messages(
            standalone, route, sid, p, exclude_ids
        )
        messages = ([rag_msgs[0]] + history + [rag_msgs[1]]) if rag_msgs else []
        if current_ids:
            _save_ctx(sid, standalone, filters, seen_ids=current_ids, last_max_salary=max_sal)
    else:
        messages = (
            [{"role": "system", "content": SYSTEM_CHAT}]
            + history[-2:]
            + [{"role": "user", "content": query}]
        )

    if not messages:
        kw, f  = _parse_query(standalone)
        answer = _no_result_reply(kw, f)
    else:
        cfg    = _ROUTE_LLM_CONFIG.get(route, _DEFAULT_LLM_CONFIG)
        answer = p["llm"].chat(messages, **cfg).choices[0].message.content

    if valid_links:
        answer = _filter_links(answer, valid_links)
    answer = _ensure_format(answer, route)

    _save_history(p, sid, query, answer)

    return jsonify({
        "content":    answer,
        "route":      route,
        "confidence": round(conf, 3),
    })


# ── RAG Helpers ───────────────────────────────────────────────────────────────

def _smart_rewrite(query: str, sid: str, route: str, p: dict, history: list) -> str:
    if route == "chitchat":
        return query
    ctx = _get_ctx(sid)
    if not ctx.get("keyword"):
        return query
    kw, _ = _parse_query(query)
    if len(kw) >= 4:
        return query

    print(f"[Rewrite] Calling LLM for: '{query}'")
    rewritten = p["reflection"].process_query(sid, query, history)
    rw_kw, _ = _parse_query(rewritten)
    if len(rw_kw) < 3:
        if ctx.get("keyword") and len(ctx["keyword"]) >= 3:
            return f"{ctx['keyword']} {query}"
        return query
    return rewritten


def _build_rag_messages(
    query: str, route: str, sid: str, p: dict, exclude_ids: list
) -> tuple[list, set, list, float, dict]:
    """Trả về (messages, valid_links, current_ids, max_salary, filters)."""
    new_kw, new_f = _parse_query(query)
    keyword, filters = _merge_ctx(sid, query, new_kw, new_f)
    if not keyword:
        keyword = query

    print(f"[RAG] kw='{keyword}' | filters={filters} | exclude={len(exclude_ids)}")
    query_vec = p["embedding"].get_query_embedding(keyword)

    jobs = p["rag"].hybrid_search(keyword, query_vec, filters=filters, limit=5, exclude_ids=exclude_ids)
    if not jobs:
        return [], set(), [], 0.0, filters

    current_ids = [j.get("job_id") for j in jobs if j.get("job_id")]
    max_sal     = max((j.get("salary_max") or 0.0 for j in jobs), default=0.0)

    context, links = p["rag"].enhance_prompt(keyword, query_vec, filters=filters, exclude_ids=exclude_ids, jobs=jobs)

    if filters.get("link_only"):
        system = SYSTEM_LINK_ONLY
    elif route == "career_advice":
        system = SYSTEM_ADVICE_RAG
    else:
        system = SYSTEM_JOB_RAG

    # ── Build filter hint block ────────────────────────────────────────────
    filter_hints: list[str] = []
    if filters.get("experience_norm") == "fresher":
        filter_hints.append("⚠️ USER LÀ FRESHER → CHỈ hiển thị job không yêu cầu KN hoặc ≤ 1 năm.")
    elif filters.get("experience_norm") == "junior":
        filter_hints.append("⚠️ USER LÀ JUNIOR → CHỈ hiển thị job yêu cầu ≤ 2 năm KN.")
    elif filters.get("experience_norm") == "senior":
        filter_hints.append("⚠️ USER LÀ SENIOR → ưu tiên job yêu cầu 3+ năm KN.")

    sal_min = filters.get("salary_min", 0)
    if sal_min > 0:
        filter_hints.append(f"⚠️ USER YÊU CẦU LƯƠNG ≥ {sal_min} triệu → CHỈ hiển thị job có lương ≥ {sal_min} triệu.")

    loc = filters.get("location_norm")
    if loc and loc not in ("other", ""):
        filter_hints.append(f"⚠️ USER YÊU CẦU ĐỊA ĐIỂM: {loc} → CHỈ hiển thị job tại {loc}.")

    hint_block = "\n".join(filter_hints)
    hint_section = f"\n\n[BỘ LỌC USER YÊU CẦU]\n{hint_block}" if hint_block else ""

    link_block   = "\n".join(f"- {u}" for u in links)
    user_content = (
        f"Câu hỏi: {query}{hint_section}\n\n"
        f"[DỮ LIỆU VIỆC LÀM]\n{context}\n\n"
        f"[LINK HỢP LỆ]\n{link_block}\n\n"
        "Chỉ dùng dữ liệu trên. KHÔNG bịa đặt. KHÔNG lặp lại job trùng nhau. Tối đa 3 job."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ], links, current_ids, max_sal, filters


def _no_result_reply(searched_kw: str, filters: dict) -> str:
    hint  = f" cho '{searched_kw}'" if searched_kw and searched_kw.lower() not in ["tìm việc", "tìm job", "hello"] else ""
    hint += f" tại {filters.get('location_norm', '')}" if filters.get("location_norm") else ""
    hint += " cho fresher" if filters.get("experience_norm") == "fresher" else ""
    return f"Mình chưa tìm thấy việc làm phù hợp{hint}.\n\nThử:\n- Mở rộng địa điểm\n- Giảm yêu cầu lương\n- Dùng từ khoá khác"


def _filter_links(text: str, valid_links: set) -> str:
    url_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    def _check(m):
        url = m.group(2).split("?")[0]
        return m.group(0) if url in valid_links else m.group(1)
    return url_pattern.sub(_check, text)


def _ensure_format(text: str, route: str) -> str:
    # Xoá markdown headings
    text = re.sub(r"#{1,6}\s*", "", text)
    if route not in ("job_search", "career_advice"):
        return text.strip()
    # Xoá các pattern LLM hay phá format
    _JUNK_PATTERNS = [
        r"Dưới đây là .*?:[ ]*\n?",
        r"Dựa trên .*?:[ ]*\n?",
        r"Mình gợi ý .*?:[ ]*\n?",
        r"Theo dữ liệu .*?:[ ]*\n?",
        r"(?m)^\s*(?:FORMAT|SAU DANH SÁCH|QUY TẮC|BỘ LỌC|LINK HỢP LỆ)[:\s]*$",
    ]
    for pat in _JUNK_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()


def _save_history(p: dict, sid: str, query: str, answer: str):
    _hist_cache.push(sid, "user", query)
    _hist_cache.push(sid, "assistant", answer)

    def worker():
        try:
            p["reflection"].save_turn(sid, query, "human")
            p["reflection"].save_turn(sid, answer, "ai")
        except Exception as e:
            print(f"[save_history] {e}")

    threading.Thread(target=worker, daemon=True).start()


# ── Shutdown ──────────────────────────────────────────────────────────────────

@atexit.register
def _shutdown():
    global _pipeline
    if _pipeline:
        for key in ["llm", "embedding"]:
            try:
                _pipeline[key].close()
            except Exception:
                pass


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    print(f"[Server] Starting Career Bot v6.3 on port {port}...")

    try:
        delete_expired(dry_run=False)
    except Exception as e:
        print(f"[Cleaner] Error: {e}")

    try:
        get_pipeline()
    except Exception as e:
        print(f"[WARNING] Warmup failed: {e}")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)