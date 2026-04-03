"""
flask_serve.py — Career Bot API Server v6
==========================================
Entrypoint chính cho Phase 2 (chạy local).

Routes:
  POST /api/v1/chat            — Chat chính (job_search, career_advice, chitchat)
  POST /api/v1/career_guidance — Tư vấn định hướng (intake 3 bước)
  GET  /api/v1/health          — Health check
  GET  /api/v1/skills          — Top skills từ taxonomy
  POST /api/v1/reset_guidance  — Reset profile session

v6 thay đổi so với v5:
  [V6-1] Schema v3: job_id, title, salary_raw, company, level
  [V6-2] Salary filter đơn vị triệu (không phải VND)
  [V6-3] Hiển thị company + level trong job format
  [V6-4] Session context TTL 2 giờ + cleanup thread
  [V6-5] _parse_query() salary: trả TRIỆU
  [V6-6] career_guidance: intake 3 bước hoàn chỉnh
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

from embedding_model.core      import EmbeddingModel
from hf_client                 import HFClient
from rag.core                  import RAG, norm_experience
from reflection.core           import SelfReflection
from semantic_router.router    import SemanticRouter
from career_guidance           import (
    SkillExtractor, MarketAnalyzer, CareerAdvisor,
    detect_field, _parse_skills_from_input, _detect_experience_level,
)
from career_guidance.advisor   import SYSTEM_GUIDANCE

app = Flask(__name__)
CORS(app)


# ── System Prompts ────────────────────────────────────────────────────────────

SYSTEM_JOB = """\
Bạn là Career Bot của TopCV — trợ lý tìm việc cho sinh viên Việt Nam.

GIỚI HẠN: Tối đa 3 việc làm. Mỗi việc 6 dòng. KHÔNG liệt kê quá 3.
Tuyệt đối KHÔNG dùng Markdown Heading (#, ##, ###).

QUY TẮC:
1. CHỈ dùng thông tin từ [DỮ LIỆU].
2. Nếu [DỮ LIỆU] không phù hợp, trả lời "Không tìm thấy công việc phù hợp".
3. Nếu user là Fresher, chỉ suggest job fresher/không yêu cầu kinh nghiệm.
4. Nếu user xin job vô lý (lương 1 tỷ/tháng...), từ chối trực tiếp.
5. QUAN TRỌNG: Nếu dữ liệu KHÔNG CÓ tên công ty hoặc có tên là "N/A", bạn TUYỆT ĐỐI KHÔNG được in ra chữ "N/A", "Chưa xác định". Bạn phải LƯỢC BỎ hoàn toàn dòng "Công ty".

FORMAT mỗi job (BẮT BUỘC):
📌 **[Tên vị trí]** — [Lương nếu có]
🏢 [Tên Công ty - KHÔNG CÓ tên công ty hoặc có tên là "N/A" thì xóa luôn dòng này, không được để chữ N/A]
📍 [Địa điểm nếu có] | 🎯 [Cấp bậc nếu có]
🎓 [Kinh nghiệm yêu cầu nếu có]
📝 [1 câu mô tả ngắn gọn nhất về công việc]
🔗 [Chi tiết công việc](link)
---

Cuối: 1 câu nhận xét trung thực về sự phù hợp.\
"""

SYSTEM_LINK_ONLY = """\
Bạn là trợ lý TopCV. Người dùng chỉ muốn xin link ứng tuyển.
Liệt kê tối đa 3 job phù hợp nhất dưới dạng:
- **[Tên vị trí]** [— Tên Công ty (chỉ ghi nếu có dữ liệu công ty)]: [link]
Không giải thích, không mô tả dài dòng.\
"""

SYSTEM_ADVICE = """\
Bạn là Career Bot của TopCV — chuyên gia phân tích kỹ năng và tư vấn hướng nghiệp.

GIỚI HẠN: Tối đa 400 từ. Trình bày rõ ràng, súc tích.

MỤC TIÊU VÀ QUY TẮC:
1. Đọc "Câu hỏi" và đoạn hội thoại trước để nhận diện tình trạng hiện tại.
2. So sánh kỹ năng người dùng với yêu cầu thực tế từ [DỮ LIỆU VIỆC LÀM].
3. Nếu có [MARKET INSIGHT], dùng số liệu từ đó để tăng tính thuyết phục.
4. KHÔNG BỊA ĐẶT kỹ năng hay mức lương ngoài dữ liệu được cung cấp.

NỘI DUNG TRẢ LỜI:
1. 【Đánh giá & Phân tích】
2. 【Lộ trình học tập】: 3 kỹ năng quan trọng nhất
3. 【Mức lương & Cơ hội】: Từ dữ liệu thực tế
4. 【Việc làm tham khảo】: Tối đa 3 job. Format: **[Tên vị trí]** [— Tên Công ty (chỉ ghi nếu có dữ liệu)]: [link]\
"""

SYSTEM_CHAT = "Bạn là Career Bot của TopCV. Trả lời ngắn gọn tiếng Việt, tối đa 3 câu."

_MAX_TOKENS = {
    "job_search":      1000,
    "career_advice":   1400,
    "career_guidance": 1600,
    "chitchat":         300,
}


# ── Pipeline (singleton) ──────────────────────────────────────────────────────

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

        embedding = EmbeddingModel()
        llm       = HFClient()
        rag       = RAG(embedding_model=embedding)
        reflection = SelfReflection(llm=llm)
        router    = SemanticRouter(embedding_model=embedding)

        # Career guidance components
        qdrant_client = rag.client
        collection    = rag.collection_name
        extractor     = SkillExtractor(qdrant_client, collection)
        market        = MarketAnalyzer(extractor).load()
        advisor       = CareerAdvisor(market, rag, llm)

        _pipeline = {
            "embedding":  embedding,
            "llm":        llm,
            "rag":        rag,
            "reflection": reflection,
            "router":     router,
            "market":     market,
            "advisor":    advisor,
        }

        print(f"[Pipeline] Ready in {time.time()-t0:.1f}s")
        return _pipeline


# ── Session Context ───────────────────────────────────────────────────────────

_CTX_LOCK:  threading.Lock = threading.Lock()
_CTX_STORE: dict[str, dict] = {}
_CTX_TTL = 7200  # 2 giờ


def _get_ctx(sid: str) -> dict:
    with _CTX_LOCK:
        return _CTX_STORE.get(sid, {})


def _save_ctx(sid: str, keyword: str, filters: dict,
              seen_ids: list | None = None, last_max_salary: float = 0.0):
    with _CTX_LOCK:
        prev = _CTX_STORE.get(sid, {})
        merged_ids = list(set(prev.get("seen_ids", []) + (seen_ids or [])))
        _CTX_STORE[sid] = {
            "keyword":         keyword,
            "filters":         filters,
            "seen_ids":        merged_ids[-30:],  # giữ tối đa 30 job_id đã xem
            "last_max_salary": last_max_salary,
            "ts":              time.time(),
        }


def _cleanup_ctx():
    """Xóa session context hết hạn. Chạy mỗi 30 phút."""
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


# ── Profile Store (career_guidance intake) ────────────────────────────────────

_PROFILE_LOCK:  threading.Lock = threading.Lock()
_PROFILE_STORE: dict[str, dict] = {}

_STEP_MAJOR  = "ask_major"
_STEP_SKILLS = "ask_skills"
_STEP_TARGET = "ask_target"
_STEP_DONE   = "complete"

_INTAKE_QUESTIONS = {
    _STEP_MAJOR: (
        "Chào bạn! Để tư vấn định hướng chính xác dựa trên nhu cầu thực tế của thị trường, "
        "mình cần biết một vài thông tin.\n\n"
        "**Câu 1/3:** Bạn đang học ngành gì, hoặc chuyên môn hiện tại của bạn là gì?\n"
        "_(Ví dụ: Công nghệ thông tin, Kinh tế, Thiết kế đồ họa, Tự học lập trình...)_"
    ),
    _STEP_SKILLS: (
        "**Câu 2/3:** Bạn hiện có những kỹ năng / công nghệ nào?\n"
        "_(Ví dụ: Python, HTML/CSS, Excel, Photoshop, hoặc 'chưa có gì cả' cũng không sao!)_"
    ),
    _STEP_TARGET: (
        "**Câu 3/3:** Bạn muốn hướng tới vị trí / lĩnh vực nào?\n"
        "_(Ví dụ: Frontend Developer, Data Analyst, UX Designer, hoặc 'chưa biết'...)_"
    ),
}


def _get_profile(sid: str) -> dict:
    with _PROFILE_LOCK:
        return dict(_PROFILE_STORE.get(sid, {}))


def _save_profile(sid: str, updates: dict):
    with _PROFILE_LOCK:
        if sid not in _PROFILE_STORE:
            _PROFILE_STORE[sid] = {}
        _PROFILE_STORE[sid].update(updates)
        _PROFILE_STORE[sid]["ts"] = time.time()


def _reset_profile(sid: str):
    with _PROFILE_LOCK:
        _PROFILE_STORE.pop(sid, None)


# ── Query Parser ──────────────────────────────────────────────────────────────

_LOCATION_KEYWORDS = {
    "hà nội": "ha_noi", "hanoi": "ha_noi", "hn": "ha_noi",
    "hồ chí minh": "ho_chi_minh", "hcm": "ho_chi_minh",
    "tp.hcm": "ho_chi_minh", "sài gòn": "ho_chi_minh",
    "đà nẵng": "da_nang", "đà nẵng": "da_nang",
    "cần thơ": "can_tho", "hải phòng": "hai_phong",
    "bình dương": "binh_duong", "đồng nai": "dong_nai",
    "remote": "remote", "work from home": "remote", "wfh": "remote",
}

_SALARY_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:[-–đến]\s*(\d+(?:[.,]\d+)?))?\s*(triệu|tr|million|m)",
    re.IGNORECASE,
)

_LINK_ONLY_KW = ["link", "url", "ứng tuyển", "apply", "xin link"]

_FRESHER_KW   = ["fresher", "mới ra trường", "chưa có kinh nghiệm", "sinh viên", "intern"]
_JUNIOR_KW    = ["junior", "1 năm", "2 năm", "mới đi làm"]
_SENIOR_KW    = ["senior", "lead", "3 năm", "4 năm", "5 năm", "nhiều năm"]


def _parse_query(query: str) -> tuple[str, dict]:
    """
    Parse query → (keyword, filters).

    filters: {
        location_norm: str | None,
        salary_min:    float (triệu) | None,  [V6-2]
        experience_norm: str | None,
        link_only:     bool,
    }
    """
    q      = query.lower()
    filters: dict = {}

    # Location
    for kw, norm in _LOCATION_KEYWORDS.items():
        if kw in q:
            filters["location_norm"] = norm
            break

    # Salary [V6-2] — đơn vị TRIỆU
    m = _SALARY_RE.search(q)
    if m:
        lo = float(m.group(1).replace(",", "."))
        hi = float(m.group(2).replace(",", ".")) if m.group(2) else lo
        # Nếu số lớn (> 1000) → đơn vị nghìn hoặc VND → chia
        if lo > 500:
            lo /= 1_000_000
        if hi > 500:
            hi /= 1_000_000
        filters["salary_min"] = round(min(lo, hi), 1)

    # Experience
    if any(k in q for k in _FRESHER_KW):
        filters["experience_norm"] = "fresher"
    elif any(k in q for k in _JUNIOR_KW):
        filters["experience_norm"] = "junior"
    elif any(k in q for k in _SENIOR_KW):
        filters["experience_norm"] = "senior"

    # Link only
    if any(k in q for k in _LINK_ONLY_KW):
        filters["link_only"] = True

    # Keyword: xóa location/salary khỏi query để làm keyword tìm kiếm
    keyword = query
    for kw in _LOCATION_KEYWORDS:
        keyword = keyword.replace(kw, "")
    keyword = _SALARY_RE.sub("", keyword)
    keyword = re.sub(r"\s+", " ", keyword).strip()

    return keyword, filters


def _merge_ctx(sid: str, query: str, new_kw: str, new_f: dict) -> tuple[str, dict]:
    """
    Merge context từ session trước với filters mới.
    - Keyword: dùng new_kw nếu đủ rõ, không thì fallback sang ctx cũ
    - Filters: merge (new_f override ctx cũ)
    - Salary "cao hơn": tăng từ last_max_salary

    [V6-3] Salary comparison đơn vị triệu.
    """
    ctx = _get_ctx(sid)
    if not ctx:
        return new_kw or query, new_f

    # Merge filters
    merged_f = dict(ctx.get("filters", {}))
    merged_f.update({k: v for k, v in new_f.items() if v is not None})

    # Salary "cao hơn" / "tốt hơn"
    if any(k in query.lower() for k in ["cao hơn", "tốt hơn", "lương cao", "nhiều hơn"]):
        last_max = ctx.get("last_max_salary", 0.0)
        if last_max > 0:
            merged_f["salary_min"] = round(last_max * 1.1, 1)  # +10%
            print(f"[Merge] salary_min nâng lên {merged_f['salary_min']} triệu")

    # Keyword
    if len(new_kw) >= 4:
        merged_kw = new_kw
    else:
        merged_kw = ctx.get("keyword", new_kw or query)

    return merged_kw, merged_f


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/api/v1/health", methods=["GET"])
def health():
    p = get_pipeline()
    return jsonify({
        "status":         "ok",
        "rag_count":      p["rag"].collection_count(),
        "bm25_status":    p["rag"].bm25_status(),
        "skills_loaded":  len(p["market"]._taxonomy) if p.get("market") else 0,
    })


@app.route("/api/v1/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or data.get("message") or "").strip()
    sid   = (data.get("session_id") or "default").strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        return _handle(query, sid)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/career_guidance", methods=["POST"])
def career_guidance_endpoint():
    data  = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    sid   = (data.get("session_id") or "default").strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        p = get_pipeline()
        return _handle_career_guidance(query, sid, p, conf=1.0)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/skills", methods=["GET"])
def skills_endpoint():
    field = request.args.get("field", "")
    n     = min(int(request.args.get("n", 20)), 50)
    try:
        p = get_pipeline()
        if field:
            skills = p["market"].top_skills_for_field(field, n=n)
        else:
            skills = p["market"].top_skills_overall(n=n)
        return jsonify({"skills": skills, "field": field or "all", "count": len(skills)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/reset_guidance", methods=["POST"])
def reset_guidance():
    sid = (request.get_json(force=True) or {}).get("session_id", "")
    if sid:
        _reset_profile(sid)
    return jsonify({"status": "profile_reset"})


# ── Core Handler ──────────────────────────────────────────────────────────────

def _handle(query: str, sid: str):
    p = get_pipeline()

    route, conf = p["router"].guide(query)
    print(f"[Handler] route={route} | conf={conf:.2f}")

    # Career guidance → intake flow
    if route == "career_guidance":
        return _handle_career_guidance(query, sid, p, conf)

    # Smart query rewrite (chỉ khi cần)
    standalone = _smart_rewrite(query, sid, route, p)

    valid_links = set()
    history     = p["reflection"].get_history(sid)

    if route in ("job_search", "career_advice"):
        ctx         = _get_ctx(sid)
        exclude_ids = ctx.get("seen_ids", [])
        rag_msgs, valid_links, current_ids, max_sal = _build_rag_messages(
            standalone, route, sid, p, exclude_ids
        )
        messages = ([rag_msgs[0]] + history + [rag_msgs[1]]) if rag_msgs else []
        if current_ids:
            _save_ctx(sid, standalone, _parse_query(standalone)[1],
                      seen_ids=current_ids, last_max_salary=max_sal)
    else:
        # chitchat
        messages = (
            [{"role": "system", "content": SYSTEM_CHAT}]
            + history
            + [{"role": "user", "content": query}]
        )

    if not messages:
        answer = _no_result_reply(sid)
    else:
        answer = p["llm"].chat(
            messages, max_tokens=_MAX_TOKENS.get(route, 800)
        ).choices[0].message.content

    if valid_links:
        answer = _filter_links(answer, valid_links)
    answer = _ensure_format(answer, route)

    _save_history(p, sid, query, answer)

    return jsonify({
        "content":    answer,
        "route":      route,
        "confidence": round(conf, 3),
    })


# ── Career Guidance Handler ───────────────────────────────────────────────────

def _handle_career_guidance(query: str, sid: str, p: dict, conf: float):
    profile = _get_profile(sid)
    step    = profile.get("step")

    # Cho phép reset giữa chừng
    if any(w in query.lower() for w in ["bắt đầu lại", "reset", "làm mới", "thay đổi thông tin"]):
        _reset_profile(sid)
        answer = _INTAKE_QUESTIONS[_STEP_MAJOR]
        _save_profile(sid, {"step": _STEP_MAJOR})
        _save_history(p, sid, query, answer)
        return jsonify({"content": answer, "route": "career_guidance", "confidence": round(conf, 3)})

    # Bước đầu: hỏi ngành
    if not step:
        answer = _INTAKE_QUESTIONS[_STEP_MAJOR]
        _save_profile(sid, {"step": _STEP_MAJOR})
        _save_history(p, sid, query, answer)
        return jsonify({"content": answer, "route": "career_guidance", "confidence": round(conf, 3)})

    if step == _STEP_MAJOR:
        _save_profile(sid, {"major": query.strip(), "step": _STEP_SKILLS})
        answer = _INTAKE_QUESTIONS[_STEP_SKILLS]
        _save_history(p, sid, query, answer)
        return jsonify({"content": answer, "route": "career_guidance", "confidence": round(conf, 3)})

    if step == _STEP_SKILLS:
        skills = _parse_skills_from_input(query)
        exp    = _detect_experience_level(profile.get("major", "") + " " + query)
        _save_profile(sid, {
            "current_skills":   skills,
            "experience_level": exp,
            "step":             _STEP_TARGET,
        })
        answer = _INTAKE_QUESTIONS[_STEP_TARGET]
        _save_history(p, sid, query, answer)
        return jsonify({"content": answer, "route": "career_guidance", "confidence": round(conf, 3)})

    if step == _STEP_TARGET:
        _save_profile(sid, {"target_role": query.strip(), "step": _STEP_DONE})
        profile = _get_profile(sid)
        return _run_guidance_analysis(query, sid, profile, p, conf)

    if step == _STEP_DONE:
        # User có thể update target_role
        if any(w in query.lower() for w in ["tôi muốn", "tư vấn", "phân tích", "roadmap", "lộ trình"]):
            _save_profile(sid, {"target_role": query.strip()})
            profile = _get_profile(sid)
        return _run_guidance_analysis(query, sid, profile, p, conf)

    # Fallback
    answer = _INTAKE_QUESTIONS[_STEP_MAJOR]
    _save_profile(sid, {"step": _STEP_MAJOR})
    _save_history(p, sid, query, answer)
    return jsonify({"content": answer, "route": "career_guidance", "confidence": round(conf, 3)})


def _run_guidance_analysis(query: str, sid: str, profile: dict, p: dict, conf: float):
    print(f"[Guidance] Running analysis | profile={profile}")

    guidance = p["advisor"].generate_guidance(
        profile, query, embedding_model=p["embedding"]
    )

    history  = p["reflection"].get_history(sid)
    messages = (
        [{"role": "system", "content": guidance["system"]}]
        + history
        + [{
            "role": "user",
            "content": (
                f"Câu hỏi: {query}\n\n"
                f"{guidance['context']}\n\n"
                "[LINK HỢP LỆ]\n"
                + "\n".join(f"- {u}" for u in guidance["valid_links"])
                + "\n\nDựa vào dữ liệu thực tế trên, hãy tư vấn định hướng nghề nghiệp "
                  "cá nhân hóa. Dùng số liệu cụ thể. Tối đa 500 từ."
            ),
        }]
    )

    answer = p["llm"].chat(
        messages, max_tokens=_MAX_TOKENS["career_guidance"]
    ).choices[0].message.content

    if guidance["valid_links"]:
        answer = _filter_links(answer, guidance["valid_links"])
    answer = _ensure_format(answer, "career_guidance")

    # Thêm coverage hint nếu thấp
    gap = guidance.get("gap", {})
    if gap and gap.get("coverage_pct", 100) < 70 and "coverage" not in answer.lower():
        missing_top3 = [s["skill"] for s in gap.get("missing_skills", [])[:3]]
        if missing_top3:
            answer += (
                f"\n\n📊 _Market data: coverage hiện tại **{gap['coverage_pct']}%**. "
                f"Ưu tiên học ngay: {', '.join(missing_top3)}_"
            )

    _save_history(p, sid, query, answer)
    return jsonify({
        "content":    answer,
        "route":      "career_guidance",
        "confidence": round(conf, 3),
        "gap_data": {
            "field":         guidance.get("field", ""),
            "coverage_pct":  gap.get("coverage_pct", 0),
            "missing_count": len(gap.get("missing_skills", [])),
        },
    })


# ── RAG Pipeline Helpers ──────────────────────────────────────────────────────

def _smart_rewrite(query: str, sid: str, route: str, p: dict) -> str:
    """Chỉ rewrite khi cần (query quá ngắn + có context cũ)."""
    if route == "chitchat":
        return query
    ctx = _get_ctx(sid)
    if not ctx.get("keyword"):
        return query
    kw, _ = _parse_query(query)
    if len(kw) >= 4:
        return query  # query đủ rõ
    print(f"[Rewrite] Calling LLM for: '{query}'")
    rewritten = p["reflection"].process_query(sid, query)
    rw_kw, _  = _parse_query(rewritten)
    if len(rw_kw) < 3:
        fallback = f"{ctx['keyword']} {query}"
        print(f"[Rewrite] Fallback: '{fallback}'")
        return fallback
    return rewritten


def _build_rag_messages(
    query: str,
    route: str,
    sid: str,
    p: dict,
    exclude_ids: list,
) -> tuple[list, set, list, float]:
    """Build [system_msg, user_msg] với RAG context. Trả empty list nếu không có jobs."""
    new_kw, new_f    = _parse_query(query)
    keyword, filters = _merge_ctx(sid, query, new_kw, new_f)
    if not keyword:
        keyword = query

    print(f"[RAG] kw='{keyword}' | filters={filters} | exclude={len(exclude_ids)}")
    query_vec = p["embedding"].get_query_embedding(keyword)

    jobs = p["rag"].hybrid_search(
        keyword, query_vec, filters=filters, limit=5, exclude_ids=exclude_ids
    )
    if not jobs:
        return [], set(), [], 0.0

    current_ids = [j["job_id"] for j in jobs if j.get("job_id")]

    # [V6-2] max_sal đơn vị triệu
    max_sal = max((j.get("salary_max") or 0.0 for j in jobs), default=0.0)

    # Market insight cho career_advice
    market_insight = ""
    if route == "career_advice":
        field = detect_field(keyword)
        if field:
            try:
                market_insight = "\n\n" + p["advisor"].quick_market_insight(field)
            except Exception:
                pass

    context, links = p["rag"].enhance_prompt(
        keyword, query_vec, filters=filters, exclude_ids=exclude_ids, jobs=jobs
    )

    # System prompt selection
    system = SYSTEM_JOB
    if filters.get("link_only"):
        system = SYSTEM_LINK_ONLY
    elif route == "career_advice":
        system = SYSTEM_ADVICE

    # Experience hint
    hint = ""
    exp = filters.get("experience_norm")
    if exp == "fresher":
        hint = "\n⚠️ USER LÀ FRESHER: CHỈ hiển thị job không yêu cầu KN hoặc dưới 1 năm."
    elif exp == "junior":
        hint = "\n[USER: 1-2 năm KN — ưu tiên job junior]"

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Câu hỏi: {query}{hint}\n\n"
                f"[DỮ LIỆU VIỆC LÀM]\n{context}"
                f"{market_insight}\n\n"
                "[LINK HỢP LỆ]\n"
                + "\n".join(f"- {u}" for u in links)
                + "\n\nChỉ dùng dữ liệu trên. KHÔNG bịa đặt. Tối đa 3 job."
            ),
        },
    ], links, current_ids, max_sal


def _no_result_reply(sid: str) -> str:
    ctx  = _get_ctx(sid)
    kw   = ctx.get("keyword", "")
    loc  = ctx.get("filters", {}).get("location_norm", "")
    exp  = ctx.get("filters", {}).get("experience_norm", "")
    hint = f" về '{kw}'" if kw else ""
    hint += f" tại {loc}" if loc else ""
    hint += " cho fresher" if exp == "fresher" else ""
    return (
        f"Mình chưa tìm thấy việc làm phù hợp{hint}.\n\n"
        "Thử:\n"
        "- Mở rộng địa điểm (bỏ bộ lọc thành phố)\n"
        "- Giảm yêu cầu lương\n"
        "- Dùng từ khoá khác (VD: 'software developer' thay vì tên công nghệ cụ thể)"
    )


def _filter_links(text: str, valid_links: set) -> str:
    """Xóa link giả trong response — chỉ giữ link có trong valid_links."""
    # Tìm tất cả URL markdown trong response
    url_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    def _check(m):
        url   = m.group(2).split("?")[0]
        label = m.group(1)
        if url in valid_links:
            return m.group(0)
        # Link không hợp lệ → để lại label nhưng xóa link
        return label
    return url_pattern.sub(_check, text)


def _ensure_format(text: str, route: str) -> str:
    """Xóa heading markdown. Kiểm tra emoji job format."""
    text = re.sub(r"#{1,6}\s*", "", text)
    if route not in ("job_search", "career_advice"):
        return text.strip()
    # Xóa dòng dẫn thừa
    text = re.sub(r"Dưới đây là .*?:\n?", "", text).strip()
    return text


def _save_history(p: dict, sid: str, query: str, answer: str):
    try:
        p["reflection"].save_turn(sid, query,  "human")
        p["reflection"].save_turn(sid, answer, "ai")
    except Exception as e:
        print(f"[save_history] {e}")


# ── Shutdown ──────────────────────────────────────────────────────────────────

@atexit.register
def _shutdown():
    global _pipeline
    if _pipeline:
        try:
            _pipeline["llm"].close()
        except Exception:
            pass
        try:
            _pipeline["embedding"].close()
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    print(f"[Server] Starting Career Bot v6 on port {port}...")
    get_pipeline()  # warm up
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
