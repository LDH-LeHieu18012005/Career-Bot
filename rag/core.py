"""
rag/core.py — Hybrid Search Pipeline v6
=========================================
[V6.1-FIX] Chuẩn hóa đầu ra company = "N/A":
  - _clean_na(): helper chuẩn hóa field rỗng/N/A
  - enhance_prompt(): bỏ dòng "Công ty" nếu N/A, không truyền N/A vào LLM
  - _rerank(): bỏ "N/A" trong pair gửi cross-encoder
"""

import os
import time
import unicodedata
import pickle
import threading
from datetime import datetime, timezone

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, Filter, FieldCondition,
    MatchValue, MatchAny, Range,
    TextIndexParams, TextIndexType, TokenizerType,
    PayloadSchemaType,
)

load_dotenv()

QDRANT_URL        = os.getenv("QDRANT_URL", "")
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION", "topcv_jobs_v3")
DEFAULT_LIMIT     = int(os.getenv("DEFAULT_SEARCH_LIMIT", 10))
SIMILARITY_THRESH = float(os.getenv("SIMILARITY_THRESHOLD", 0.25))

CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "jinaai/jina-reranker-v3",
)
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 10))

_BM25_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "bm25_cache.pkl")
)


# ── Section → field mapping (dùng trong BM25 và _load_full_jobs) ───────────────
_SECTION_TO_FIELD = {
    "description":  "mo_ta",
    "requirements": "yeu_cau",
    "benefits":     "quyen_loi",
}

# ── Experience normalization ──────────────────────────────────────────────────

def norm_experience(experience: str, level: str) -> str:
    lvl = (level    or "").lower().strip()
    exp = (experience or "").lower().strip()

    if "fresher" in lvl:
        return "fresher"
    if "junior" in lvl:
        return "junior"
    if any(k in lvl for k in ["senior", "lead", "manager", "principal"]):
        return "senior"

    if any(k in exp for k in ["không yêu cầu", "chưa có", "0 năm", "dưới 1"]):
        return "fresher"
    if any(f"{i} năm" in exp for i in [1, 2]):
        return "junior"
    if any(f"{i} năm" in exp for i in range(3, 15)):
        return "senior"

    return "other"


# ── Vietnamese text utils ─────────────────────────────────────────────────────

def _strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    s   = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return s.replace("\u0111", "d").replace("\u0110", "D")


_STOPWORDS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "vn_stopwords.txt")
)
_VI_STOPWORDS: set[str] = set()
try:
    with open(_STOPWORDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                _VI_STOPWORDS.add(_strip_accents(w).lower())
except FileNotFoundError:
    print(f"[RAG] Warning: {_STOPWORDS_PATH} not found")
    _VI_STOPWORDS = {"tim", "kiem", "cho", "minh", "toi", "ban", "co", "khong", "duoc", "tai"}


def _tokenize(text: str) -> list[str]:
    return [
        w for w in _strip_accents(text).lower().split()
        if len(w) >= 2 and w not in _VI_STOPWORDS
    ]


# ── [V6.1-FIX] Helper chuẩn hóa field ───────────────────────────────────────

def _clean_na(value: str | None, fallback: str = "") -> str:
    """
    Trả về fallback nếu value là None, rỗng, hoặc "N/A" (case-insensitive).
    Dùng để chuẩn hóa trước khi đưa vào context/prompt — tránh LLM thấy "N/A".
    """
    if not value:
        return fallback
    stripped = value.strip()
    if stripped.upper() == "N/A" or stripped == "":
        return fallback
    return stripped


# ── RAG ───────────────────────────────────────────────────────────────────────

class RAG:

    def __init__(self, embedding_model=None, collection_name: str = COLLECTION_NAME):
        self.embedding_model       = embedding_model
        self.collection_name       = collection_name
        self._bm25                 = None
        self._bm25_corpus: list[dict] = []
        self._bm25_ready           = False
        self._cross_encoder        = None
        self._cross_encoder_ready  = False
        self._ce_lock              = threading.Lock()
        self.client                = self._connect()
        self._init_collection()
        threading.Thread(target=self._get_cross_encoder, daemon=True, name="CE-warmup").start()

    def _connect(self) -> QdrantClient:
        if QDRANT_URL:
            print(f"[RAG] Qdrant Cloud: {QDRANT_URL}")
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
        print(f"[RAG] Qdrant Local: {QDRANT_HOST}:{QDRANT_PORT}")
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)

    def _init_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            count = self.client.count(self.collection_name).count
            print(f"[RAG] Collection '{self.collection_name}' | {count:,} vectors")
            self._build_bm25()
        else:
            if self.embedding_model is None:
                raise RuntimeError(
                    f"Collection '{self.collection_name}' chưa tồn tại. "
                    "Chạy kaggle_ingest.py trước."
                )
            self._create_collection()

    def _create_collection(self):
        dim = self.embedding_model.dimension
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        content_fields = ["description", "requirements", "benefits"]
        for field in content_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name, field_name=field,
                    field_schema=TextIndexParams(
                        type=TextIndexType.TEXT, tokenizer=TokenizerType.WORD,
                        min_token_len=2, max_token_len=20, lowercase=True,
                    ),
                )
            except Exception as e:
                print(f"[RAG] Could not create text index for '{field}': {e}")

        for field in ("section", "job_id", "location_norm", "level", "group_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name, field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                print(f"[RAG] Could not create keyword index for '{field}': {e}")

        for field in ("salary_min", "salary_max", "deadline_ts"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name, field_name=field,
                    field_schema=PayloadSchemaType.FLOAT,
                )
            except Exception as e:
                print(f"[RAG] Could not create float index for '{field}': {e}")

        print(f"[RAG] Created collection '{self.collection_name}' | dim={dim}")

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[RAG] rank-bm25 chưa cài: pip install rank-bm25")
            return

        total_vectors = self.client.count(self.collection_name).count

        if os.path.exists(_BM25_CACHE_PATH):
            try:
                with open(_BM25_CACHE_PATH, "rb") as f:
                    cached = pickle.load(f)
                if (cached.get("count")      == total_vectors and
                        cached.get("collection") == self.collection_name):
                    self._bm25_corpus = cached["corpus"]
                    self._bm25        = cached["bm25"]
                    self._bm25_ready  = True
                    print(
                        f"[RAG] BM25 loaded from cache | "
                        f"{len(self._bm25_corpus):,} jobs | {total_vectors:,} vectors"
                    )
                    return
                else:
                    print("[RAG] BM25 cache stale — rebuilding...")
            except Exception as e:
                print(f"[RAG] BM25 cache invalid ({e}) — rebuilding...")

        print("[RAG] Building BM25 index...", end=" ", flush=True)
        t0 = time.time()

        jobs_map: dict[str, dict] = {}
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000, offset=offset,
                with_payload=True, with_vectors=False,
            )
            if not batch:
                break
            for point in batch:
                payload = point.payload
                job_id  = payload.get("job_id", "")
                section = payload.get("section", "")

                content_field = ""
                if section == "description":
                    content_field = payload.get("description", "") or ""
                elif section == "requirements":
                    content_field = payload.get("requirements", "") or ""
                elif section == "benefits":
                    content_field = payload.get("benefits", "") or ""
                elif section == "overview":
                    parts = [f"Vị trí: {payload.get('title', '')}"]
                    salary_val = _clean_na(payload.get("salary_raw"))
                    if salary_val and salary_val not in ("Thoả thuận",):
                        parts.append(f"Mức lương: {salary_val}")
                    if _clean_na(payload.get("experience")):
                        parts.append(f"Kinh nghiệm: {_clean_na(payload['experience'])}")
                    if _clean_na(payload.get("level")):
                        parts.append(f"Cấp bậc: {_clean_na(payload['level'])}")
                    content_field = ". ".join(part for part in parts if part).strip()

                if not content_field:
                    content_parts = [
                        payload.get("title", ""),
                        payload.get("company", ""),
                        payload.get("location", ""),
                        payload.get("experience", ""),
                        payload.get("level", ""),
                        payload.get("salary_raw", ""),
                    ]
                    content_field = " ".join(part for part in content_parts if part).strip()

                if not job_id or not content_field.strip():
                    continue

                if job_id not in jobs_map:
                    jobs_map[job_id] = {
                        "payload":      payload,
                        "texts":        [],
                        "has_overview": False,
                    }

                if section == "overview" and not jobs_map[job_id]["has_overview"]:
                    jobs_map[job_id]["payload"] = payload
                    jobs_map[job_id]["has_overview"] = True

                stripped_content = content_field.strip()
                if stripped_content:
                    jobs_map[job_id]["texts"].append(stripped_content)

            if offset is None:
                break

        corpus = []
        for job_id, job_data in jobs_map.items():
            combined_text = " ".join(job_data["texts"]).strip()
            if combined_text:
                corpus.append({
                    "payload":       job_data["payload"],
                    "combined_text": combined_text,
                })

        if not corpus:
            print(f"\n[ERROR] No valid documents found for BM25.")
            self._bm25_corpus = []
            self._bm25        = None
            self._bm25_ready  = False
            return

        corpus_tokenized = [_tokenize(item["combined_text"]) for item in corpus]

        valid_corpus_tokenized = []
        valid_corpus_items     = []
        for tokens, item in zip(corpus_tokenized, corpus):
            if tokens:
                valid_corpus_tokenized.append(tokens)
                valid_corpus_items.append(item)

        if not valid_corpus_tokenized:
            print(f"\n[ERROR] All documents resulted in empty tokens.")
            self._bm25_corpus = []
            self._bm25        = None
            self._bm25_ready  = False
            return

        self._bm25_corpus = valid_corpus_items
        self._bm25        = BM25Okapi(valid_corpus_tokenized)
        self._bm25_ready  = True

        elapsed = time.time() - t0
        print(f"{len(valid_corpus_items):,} jobs | {total_vectors:,} vectors | {elapsed:.1f}s")

        try:
            os.makedirs(os.path.dirname(_BM25_CACHE_PATH), exist_ok=True)
            with open(_BM25_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "count":      total_vectors,
                    "collection": self.collection_name,
                    "corpus":     valid_corpus_items,
                    "bm25":       self._bm25,
                }, f)
            print(f"[RAG] BM25 cache saved → {_BM25_CACHE_PATH}")
        except Exception as e:
            print(f"[RAG] BM25 cache save failed: {e}")

    def invalidate_bm25_cache(self):
        if os.path.exists(_BM25_CACHE_PATH):
            os.remove(_BM25_CACHE_PATH)
            print("[RAG] BM25 cache invalidated.")
        self._bm25_ready  = False
        self._bm25_corpus = []
        self._build_bm25()

    # ── Cross-Encoder ─────────────────────────────────────────────────────────

    def _get_cross_encoder(self):
        if self._cross_encoder_ready:
            return self._cross_encoder
        with self._ce_lock:
            if self._cross_encoder_ready:
                return self._cross_encoder
            self._cross_encoder_ready = True
            try:
                from sentence_transformers import CrossEncoder
                print(f"[RAG] Loading cross-encoder: {CROSS_ENCODER_MODEL}...", end=" ", flush=True)
                self._cross_encoder = CrossEncoder(
                    CROSS_ENCODER_MODEL, 
                    trust_remote_code=True,
                    model_kwargs={"torch_dtype": "auto"},
                    local_files_only=True
                )
                if self._cross_encoder.tokenizer.pad_token is None:
                    self._cross_encoder.tokenizer.pad_token = self._cross_encoder.tokenizer.eos_token
                print("OK")
            except Exception as e:
                print(f"[RAG] Cross-encoder failed: {e}")
            return self._cross_encoder

    # ── Filter builder ────────────────────────────────────────────────────────

    def _build_filter(
        self,
        filters: dict,
        exclude_ids: list[str] | None = None,
    ) -> Filter | None:
        must:     list = []
        must_not: list = []

        loc = filters.get("location_norm")
        if loc and loc not in ("other", "", None):
            must.append(FieldCondition(key="location_norm", match=MatchValue(value=loc)))

        sal = filters.get("salary_min") or 0.0
        if sal > 0:
            must.append(FieldCondition(key="salary_max", range=Range(gte=float(sal))))

        # [Auto-Update Filter] Loại trừ công việc deadline_ts < now
        now_ts = datetime.now(timezone.utc).timestamp()
        must.append(FieldCondition(key="deadline_ts", range=Range(gte=now_ts)))

        exp_norm = filters.get("experience_norm")
        if exp_norm == "fresher":
            must_not.extend([
                FieldCondition(key="level_norm", match=MatchAny(any=["senior", "lead", "manager", "director"])),
                FieldCondition(key="experience_norm", match=MatchAny(any=["expert"]))
            ])
        elif exp_norm == "junior":
            must_not.extend([
                FieldCondition(key="level_norm", match=MatchAny(any=["lead", "manager", "director"])),
            ])

        if exclude_ids:
            must_not.append(FieldCondition(key="job_id", match=MatchAny(any=exclude_ids)))

        if not must and not must_not:
            return None
        return Filter(must=must, must_not=must_not)

    # ── Vector Search ─────────────────────────────────────────────────────────

    def vector_search(
        self,
        query_vec: list[float],
        limit: int = DEFAULT_LIMIT,
        qdrant_filter: Filter | None = None,
    ) -> list[dict]:
        kwargs = dict(
            collection_name=self.collection_name,
            query=query_vec,
            limit=limit * 2,
            score_threshold=SIMILARITY_THRESH,
            with_payload=True,
        )
        if qdrant_filter:
            kwargs["query_filter"] = qdrant_filter

        try:
            points = self.client.query_points(**kwargs).points
        except Exception as e:
            print(f"[RAG] Vector search error: {e}")
            return []

        if not points:
            kwargs.pop("score_threshold")
            try:
                points = self.client.query_points(**kwargs).points
                if points:
                    print(f"[RAG] Vector: fallback (no threshold) → {len(points)} docs")
            except Exception:
                return []

        return [_to_job(p.payload, p.score) for p in points]

    # ── BM25 Search ───────────────────────────────────────────────────────────

    def bm25_search(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        filters: dict | None = None,
        exclude_ids: list[str] | None = None,
    ) -> list[dict]:
        if not self._bm25_ready or self._bm25 is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        if len(scores) == 0 or max(scores) <= 0:
            return []

        max_s   = max(scores)
        filters = filters or {}
        exclude = set(exclude_ids or [])

        top_idx = [
            i for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            if scores[i] > 0
        ]

        results, seen = [], set()
        now_ts = datetime.now(timezone.utc).timestamp()

        for idx in top_idx:
            item   = self._bm25_corpus[idx]
            pl     = item["payload"]
            job_id = pl.get("job_id", "")

            if job_id in exclude:
                continue

            if filters:
                loc = filters.get("location_norm")
                if loc and loc not in ("other", "", None):
                    if pl.get("location_norm") != loc:
                        continue
                sal = filters.get("salary_min") or 0.0
                if sal > 0:
                    jmax = pl.get("salary_max", 0) or 0
                    if jmax > 0 and jmax < sal:
                        continue

                # [Auto-Update Filter] Realtime check trong array bm25
                dts = pl.get("deadline_ts", 0.0)
                if dts and dts > 0 and dts < now_ts:
                    continue

            if job_id and job_id not in seen:
                seen.add(job_id)
                results.append(_to_job(pl, scores[idx] / max_s if max_s > 0 else 0.0))

            if len(results) >= limit * 2:
                break

        print(f"[RAG] BM25: {len(results)} docs | tokens={tokens[:5]}")
        return results

    # ── Load full job data ────────────────────────────────────────────────────

    def _load_full_jobs(self, job_ids: list[str]) -> list[dict]:
        if not job_ids:
            return []

        order = {jid: i for i, jid in enumerate(job_ids)}

        try:
            hits, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="job_id", match=MatchAny(any=job_ids)),
                ]),
                limit=len(job_ids) * 5,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            print(f"[RAG] Load full jobs error: {e}")
            return []

        merged: dict[str, dict] = {}
        for h in hits:
            pl     = h.payload
            job_id = pl.get("job_id", "")
            if not job_id or job_id not in order:
                continue

            if job_id not in merged:
                merged[job_id] = _to_job(pl, 1.0)
                merged[job_id]["_rrf_rank"] = order[job_id]

            section = pl.get("section", "")
            if section in _SECTION_TO_FIELD:
                field = _SECTION_TO_FIELD[section]
                merged[job_id][field] = pl.get(section, "") or ""
            elif section == "overview" and not merged[job_id].get("has_overview"):
                merged[job_id].update(_to_job(pl, 1.0))
                merged[job_id]["_rrf_rank"]    = order[job_id]
                merged[job_id]["has_overview"] = True

        jobs = sorted(merged.values(), key=lambda j: j.get("_rrf_rank", 999))
        print(f"[RAG] Loaded {len(jobs)} full jobs")
        return jobs

    # ── Hybrid Search ─────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        query_vec: list[float],
        filters: dict | None = None,
        limit: int = DEFAULT_LIMIT,
        exclude_ids: list[str] | None = None,
    ) -> list[dict]:
        filters       = filters or {}
        qdrant_filter = self._build_filter(filters, exclude_ids)

        vec_results  = self.vector_search(query_vec, limit=RERANK_TOP_K, qdrant_filter=qdrant_filter)
        bm25_results = self.bm25_search(query, limit=RERANK_TOP_K, filters=filters, exclude_ids=exclude_ids)

        print(f"[RAG] Raw: vector={len(vec_results)}, bm25={len(bm25_results)}")
        if not vec_results and not bm25_results:
            return []

        fused = _rrf([vec_results, bm25_results], weights=[0.6, 0.4])

        unique_jobs, seen_urls = [], set()
        for j in fused:
            url = j.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_jobs.append(j)
            elif not url:
                unique_jobs.append(j)

        job_ids   = [j["job_id"] for j in unique_jobs[:RERANK_TOP_K] if j.get("job_id")]
        full_jobs = self._load_full_jobs(job_ids)

        return self._rerank(query, full_jobs, top_n=limit)

    # ── [V6.1-FIX] _rerank: bỏ "N/A" company khi build pair cross-encoder ───

    def _rerank(self, query: str, jobs: list[dict], top_n: int = DEFAULT_LIMIT) -> list[dict]:
        ce = self._get_cross_encoder()
        if ce is None or not jobs:
            return sorted(jobs, key=lambda j: j.get("_rrf_rank", 999))[:top_n]

        pairs = []
        for j in jobs:
            company_str = _clean_na(j.get("company"))
            # Chỉ thêm "Công ty: ..." vào pair nếu có dữ liệu thực
            company_part = f"Công ty: {company_str}. " if company_str else ""
            pairs.append([
                query,
                f"Vị trí: {j.get('title','')}. "
                f"{company_part}"
                f"Lương: {_clean_na(j.get('salary_raw'), 'Thỏa thuận')}. "
                f"Địa điểm: {j.get('location','')}. "
                f"Cấp bậc: {_clean_na(j.get('level'))}. "
                f"Kinh nghiệm: {_clean_na(j.get('experience'), 'Không yêu cầu')}. "
                f"Yêu cầu: {(j.get('yeu_cau') or '')[:500]}"
            ])

        try:
            scores = ce.predict(pairs, batch_size=1)
            ranked = sorted(zip(jobs, scores), key=lambda x: x[1], reverse=True)
            print(f"[RAG] Rerank: top={ranked[0][1]:.3f} | n={len(ranked)}")
            return [j for j, _ in ranked[:top_n]]
        except Exception as e:
            print(f"[RAG] Rerank error: {e}")
            return jobs[:top_n]

    # ── [V6.1-FIX] enhance_prompt: không truyền "N/A" vào context LLM ────────

    def enhance_prompt(
        self,
        query: str,
        query_vec: list[float],
        filters: dict | None = None,
        exclude_ids: list[str] | None = None,
        jobs: list[dict] | None = None,
    ) -> tuple[str, set[str]]:
        if jobs is None:
            jobs = self.hybrid_search(
                query, query_vec, filters=filters, exclude_ids=exclude_ids
            )
        if not jobs:
            return "Không tìm thấy công việc phù hợp.", set()

        jobs        = jobs[:10]
        valid_links = {j.get("url", "").split("?")[0] for j in jobs if j.get("url")}
        lines       = []

        for i, j in enumerate(jobs, 1):
            clean_url = j.get("url", "").split("?")[0]
            company   = _clean_na(j.get("company"))   # ← [V6.1-FIX]
            salary    = _clean_na(j.get("salary_raw"), "Thỏa thuận")
            exp       = _clean_na(j.get("experience"), "Không yêu cầu")
            level     = _clean_na(j.get("level"))

            lines += [
                f"[JOB {i}]",
                f"Tên vị trí : {j.get('title', 'Chưa xác định')}",
            ]

            # Chỉ thêm dòng Công ty nếu có dữ liệu thực  ← [V6.1-FIX]
            if company:
                lines.append(f"Công ty    : {company}")

            lines += [
                f"Lương      : {salary}",
                f"Địa điểm   : {j.get('location', 'Chưa xác định')}",
            ]

            if level:
                lines.append(f"Cấp bậc    : {level}")

            lines.append(f"Kinh nghiệm: {exp}")

            if j.get("mo_ta"):
                lines.append(f"Mô tả      : {j['mo_ta'][:400]}")
            if j.get("yeu_cau"):
                lines.append(f"Yêu cầu    : {j['yeu_cau'][:500]}")
            if j.get("quyen_loi"):
                lines.append(f"Quyền lợi  : {j['quyen_loi'][:200]}")

            lines += [f"Link       : {clean_url}", "---"]

        return "\n".join(lines), valid_links

    # ── Utilities ─────────────────────────────────────────────────────────────

    def collection_count(self) -> int:
        return self.client.count(self.collection_name).count

    def bm25_status(self) -> dict:
        return {"ready": self._bm25_ready and self._bm25 is not None, "docs": len(self._bm25_corpus)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_job(payload: dict, score: float) -> dict:
    """Chuẩn hóa payload → job dict (schema v3)."""
    return {
        "job_id":        payload.get("job_id",       ""),
        "title":         payload.get("title",         ""),
        "company":       payload.get("company",       ""),   # giữ nguyên raw
        "salary_raw":    payload.get("salary_raw",    ""),
        "salary_min":    payload.get("salary_min",    0.0),
        "salary_max":    payload.get("salary_max",    0.0),
        "location":      payload.get("location",      ""),
        "location_norm": payload.get("location_norm", ""),
        "experience":    payload.get("experience",    ""),
        "level":         payload.get("level",         ""),
        "url":           payload.get("url",           ""),
        "deadline":      payload.get("deadline",      ""),
        "mo_ta":         payload.get("mo_ta",         ""),
        "yeu_cau":       payload.get("yeu_cau",       ""),
        "quyen_loi":     payload.get("quyen_loi",     ""),
        "score":         round(float(score), 4),
    }


def _rrf(
    doc_lists: list[list[dict]],
    weights: list[float],
    c: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion với weights."""
    all_docs: dict[str, dict] = {}
    for lst in doc_lists:
        for d in lst:
            jid = d.get("job_id", "")
            if jid and jid not in all_docs:
                all_docs[jid] = d

    scores: dict[str, float] = {k: 0.0 for k in all_docs}
    for lst, w in zip(doc_lists, weights):
        for rank, d in enumerate(lst, 1):
            jid = d.get("job_id", "")
            if jid in scores:
                scores[jid] += w / (rank + c)

    return [all_docs[k] for k in sorted(scores, key=lambda x: scores[x], reverse=True)]