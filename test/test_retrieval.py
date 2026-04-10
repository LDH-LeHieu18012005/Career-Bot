"""
eval_retrieval_recall.py — Đánh giá Recall@1/5/10 cho Hybrid Search Pipeline v6
=================================================================================
Script này **giả lập chính xác** luồng retrieval thật trong hệ thống:
  1. _parse_query()  → trích xuất keyword + filters (giống flask_serve.py)
  2. EmbeddingModel  → embed query bằng Jina v3 (task=retrieval.query)
  3. RAG.hybrid_search() → Vector + BM25 → RRF fusion → Cross-encoder rerank
  4. Tính Recall@1, Recall@5, Recall@10 theo từng query
  5. In báo cáo chi tiết và summary table

Cách chạy:
    cd career_bot_v6
    python eval_retrieval_recall.py

Hoặc chỉ định file CSV:
    python eval_retrieval_recall.py --csv test_retrieval_50q.csv
    python eval_retrieval_recall.py --csv test_retrieval_50q.csv --limit 20 --no-rerank
"""

import os
import re
import sys
import time
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

# ── Cho phép import từ thư mục gốc project ───────────────────────────────────
# File này có thể đặt trong thư mục gốc hoặc thư mục con (vd: test/)
# → tự động leo lên cho đến khi tìm thấy thư mục chứa embedding_model/
_here = Path(__file__).resolve().parent
PROJECT_ROOT = _here
for _candidate in [_here, _here.parent, _here.parent.parent]:
    if (_candidate / "embedding_model").is_dir() and (_candidate / "rag").is_dir():
        PROJECT_ROOT = _candidate
        break

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # đảm bảo .env và data/ paths đúng khi dùng relative paths

from dotenv import load_dotenv

# Load .env từ project root (phải sau khi PROJECT_ROOT được xác định ở trên)
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1: Query Parser — sao chép nguyên từ flask_serve.py
# ─────────────────────────────────────────────────────────────────────────────

_LOCATION_KEYWORDS = {
    "hà nội": "ha_noi", "hanoi": "ha_noi", "hn": "ha_noi",
    "hồ chí minh": "ho_chi_minh", "hcm": "ho_chi_minh",
    "tp.hcm": "ho_chi_minh", "sài gòn": "ho_chi_minh",
    "cần thơ": "can_tho", "hải phòng": "hai_phong",
    "bình dương": "binh_duong", "đồng nai": "dong_nai",
    "remote": "remote", "work from home": "remote", "wfh": "remote",
}

_SALARY_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:[-–đến]\s*(\d+(?:[.,]\d+)?))?\\s*(triệu|tr|million|m)",
    re.IGNORECASE,
)

_LINK_ONLY_KW = ["link", "url", "ứng tuyển", "apply", "xin link"]
_FRESHER_KW   = ["fresher", "mới ra trường", "chưa có kinh nghiệm", "sinh viên", "intern",
                  "mới học", "tự học", "chưa biết gì", "học việc", "thực tập"]
_JUNIOR_KW    = ["junior", "1 năm", "2 năm", "mới đi làm"]
_SENIOR_KW    = ["senior", "lead", "3 năm", "4 năm", "5 năm", "nhiều năm"]


def _parse_query(query: str) -> tuple[str, dict]:
    """
    Trích xuất keyword + filters từ query.
    Hàm này sao chép y chang logic trong flask_serve.py — không được thay đổi.
    """
    q = query.lower()
    filters: dict = {}

    # Location
    for kw, norm in _LOCATION_KEYWORDS.items():
        if kw in q:
            filters["location_norm"] = norm
            break

    # Salary
    m = _SALARY_RE.search(q)
    if m:
        lo = float(m.group(1).replace(",", "."))
        hi = float(m.group(2).replace(",", ".")) if m.group(2) else lo
        if lo > 500:
            lo /= 1_000_000
        if hi > 500:
            hi /= 1_000_000
        filters["salary_min"] = round(min(lo, hi), 1)

    # Experience level
    if any(k in q for k in _FRESHER_KW):
        filters["experience_norm"] = "fresher"
    elif any(k in q for k in _JUNIOR_KW):
        filters["experience_norm"] = "junior"
    elif any(k in q for k in _SENIOR_KW):
        filters["experience_norm"] = "senior"

    # Link-only flag (không ảnh hưởng retrieval)
    if any(k in q for k in _LINK_ONLY_KW):
        filters["link_only"] = True

    # Keyword cleaned
    keyword = query
    for kw in _LOCATION_KEYWORDS:
        keyword = keyword.replace(kw, "")
    keyword = _SALARY_RE.sub("", keyword)
    keyword = re.sub(r"\s+", " ", keyword).strip()

    return keyword, filters


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2: Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """
    Recall@K = |expected ∩ top-K retrieved| / |expected|
    Nếu expected_ids rỗng → trả 0.0 để tránh chia cho 0.
    """
    if not expected_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in expected_ids)
    return hits / len(expected_ids)


def hit_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> bool:
    """Hit@K = có ít nhất 1 expected_id trong top-K hay không."""
    return any(rid in expected_ids for rid in retrieved_ids[:k])


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3: Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEvaluator:

    KS = [1, 5, 10]

    def __init__(self, csv_path: str, limit: int = 10, no_rerank: bool = False):
        self.csv_path  = csv_path
        self.limit     = max(limit, max(self.KS))  # cần lấy ít nhất top-10
        self.no_rerank = no_rerank
        self.results   = []

        print("\n" + "="*65)
        print("  Career Bot v6 — Retrieval Recall Evaluation")
        print("="*65)
        print(f"  CSV       : {csv_path}")
        print(f"  Limit     : {self.limit}")
        print(f"  Rerank    : {'DISABLED' if no_rerank else 'ENABLED (cross-encoder)'}")
        print("="*65 + "\n")

        self._load_models()
        self._load_dataset()

    def _load_models(self):
        """Khởi tạo EmbeddingModel + RAG — giống hệt khi Flask start."""
        print("[Init] Loading EmbeddingModel (Jina v3)...")
        t0 = time.time()
        from embedding_model.core import EmbeddingModel
        self.embed_model = EmbeddingModel()
        print(f"[Init] EmbeddingModel ready ({time.time()-t0:.1f}s)\n")

        print("[Init] Connecting to Qdrant + building BM25...")
        t0 = time.time()
        from rag.core import RAG
        self.rag = RAG(embedding_model=self.embed_model)

        # Nếu --no-rerank: patch _rerank để bỏ qua cross-encoder
        if self.no_rerank:
            import types
            def _rerank_bypass(self_rag, query, jobs, top_n=10):
                return sorted(jobs, key=lambda j: j.get("_rrf_rank", 999))[:top_n]
            self.rag._rerank = types.MethodType(_rerank_bypass, self.rag)
            print("[Init] Cross-encoder rerank BYPASSED\n")
        else:
            # Warm up cross-encoder ngay (không đợi daemon thread)
            print("[Init] Pre-loading cross-encoder...")
            self.rag._get_cross_encoder()

        print(f"[Init] RAG ready ({time.time()-t0:.1f}s)\n")

    def _load_dataset(self):
        print(f"[Data] Loading {self.csv_path}...")
        df = pd.read_csv(self.csv_path)

        # Giữ job_search + cho phép career_advice nhưng xử lý riêng
        retrieval_types = {
            "job_search", 
            "job_search+filter", 
            "career_advice", 
            "career_advice+filter"
        }
        
        self.df = df[df["type"].isin(retrieval_types)].reset_index(drop=True)

        total    = len(df)
        filtered = len(self.df)
        skipped  = total - filtered

        print(f"[Data] Total queries: {total}")
        print(f"[Data] Evaluated queries: {filtered} (job_search + career_advice)")
        print(f"[Data] Skipped: {skipped}\n")

    def run(self):
        rows = self.df.to_dict("records")
        print(f"Running evaluation on {len(rows)} queries...\n")
        print("-" * 65)

        for i, row in enumerate(rows, 1):
            qid      = row["id"]
            query    = row["query"]
            expected = set(
                eid.strip() for eid in str(row["expected_job_ids"]).split("|")
                if eid.strip()
            )
            difficulty = row.get("difficulty", "?")
            qtype      = row.get("type", "?")

            # ── Bước 1: Parse query (giống flask_serve._parse_query) ──
            keyword, filters = _parse_query(query)

            # ── Bước 2: Embed query (giống EmbeddingModel.get_query_embedding) ──
            t_start = time.time()
            query_vec = self.embed_model.get_query_embedding(keyword or query)

            # ── Bước 3: Hybrid search (vector + BM25 + rerank) ──────────────
            jobs = self.rag.hybrid_search(
                query     = keyword or query,
                query_vec = query_vec,
                filters   = filters,
                limit     = self.limit,
            )
            elapsed = time.time() - t_start

            retrieved_ids = [j["job_id"] for j in jobs if j.get("job_id")]

            # ── Bước 4: Tính metrics ─────────────────────────────────────────
            rec   = {k: recall_at_k(retrieved_ids, expected, k) for k in self.KS}
            hits  = {k: hit_at_k(retrieved_ids, expected, k)    for k in self.KS}

            self.results.append({
                "id":         qid,
                "query":      query,
                "keyword":    keyword,
                "filters":    filters,
                "expected":   expected,
                "retrieved":  retrieved_ids,
                "difficulty": difficulty,
                "type":       qtype,
                "time_s":     elapsed,
                **{f"recall@{k}": rec[k] for k in self.KS},
                **{f"hit@{k}":    hits[k] for k in self.KS},
            })

            # ── In progress ─────────────────────────────────────────────────
            hit_str = " | ".join(
                f"H@{k}={'✓' if hits[k] else '✗'}" for k in self.KS
            )
            rec_str = " | ".join(
                f"R@{k}={rec[k]:.2f}" for k in self.KS
            )
            print(
                f"[{i:>3}/{len(rows)}] {qid} [{difficulty:>6}] "
                f"{hit_str} | {rec_str} | {elapsed:.2f}s"
            )
            if not hits[1] and expected:
                top3_titles = [j.get("title", "?") for j in jobs[:3]]
                print(f"         ↳ Expected: {expected}")
                print(f"         ↳ Got top3: {top3_titles}")

        self._report()

    # ── Report ────────────────────────────────────────────────────────────────

    def _report(self):
        if not self.results:
            print("\n[!] Không có kết quả để báo cáo.")
            return

        df = pd.DataFrame(self.results)
        n  = len(df)

        print("\n" + "="*65)
        print(f"  SUMMARY — {n} queries evaluated")
        print("="*65)

        # Overall
        print("\n▶ Overall Recall & Hit Rate")
        print(f"  {'Metric':<15} {'@1':>8} {'@5':>8} {'@10':>8}")
        print(f"  {'-'*43}")
        for metric in ("recall", "hit"):
            vals = [df[f"{metric}@{k}"].mean() for k in self.KS]
            print(f"  {metric.capitalize()+'@K':<15} " + "".join(f"{v:>8.3f}" for v in vals))

        # By difficulty
        print("\n▶ Recall@K by Difficulty")
        print(f"  {'Difficulty':<12} {'N':>4} {'@1':>8} {'@5':>8} {'@10':>8}")
        print(f"  {'-'*42}")
        for diff in ["easy", "medium", "hard"]:
            sub = df[df["difficulty"] == diff]
            if len(sub) == 0:
                continue
            vals = [sub[f"recall@{k}"].mean() for k in self.KS]
            print(
                f"  {diff:<12} {len(sub):>4} " +
                "".join(f"{v:>8.3f}" for v in vals)
            )

        # By type
        print("\n▶ Recall@K by Query Type")
        print(f"  {'Type':<22} {'N':>4} {'@1':>8} {'@5':>8} {'@10':>8}")
        print(f"  {'-'*52}")
        for qtype in sorted(df["type"].unique()):
            sub = df[df["type"] == qtype]
            vals = [sub[f"recall@{k}"].mean() for k in self.KS]
            print(
                f"  {qtype:<22} {len(sub):>4} " +
                "".join(f"{v:>8.3f}" for v in vals)
            )

        # Failed @10
        failed = df[df["hit@10"] == False]
        if len(failed) > 0:
            print(f"\n▶ Queries MISSED at @10 ({len(failed)} queries)")
            for _, r in failed.iterrows():
                print(f"  [{r['id']}] {r['query'][:60]}")
                print(f"         expected : {r['expected']}")
                print(f"         retrieved: {r['retrieved'][:5]}")

        # Timing
        avg_t = df["time_s"].mean()
        max_t = df["time_s"].max()
        print(f"\n▶ Latency")
        print(f"  Avg: {avg_t:.2f}s  |  Max: {max_t:.2f}s")

        # Save CSV
        ts       = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = Path(self.csv_path).parent / f"eval_results_{ts}.csv"
        save_cols = [
            "id", "query", "keyword", "filters", "difficulty", "type",
            "recall@1", "recall@5", "recall@10",
            "hit@1",    "hit@5",    "hit@10",
            "time_s", "expected", "retrieved",
        ]
        df[save_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ Kết quả chi tiết lưu tại: {out_path}")
        print("="*65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4: CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá Recall@1/5/10 cho Career Bot v6 Retrieval Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Ví dụ:
              python eval_retrieval_recall.py
              python eval_retrieval_recall.py --csv path/to/test.csv
              python eval_retrieval_recall.py --no-rerank
              python eval_retrieval_recall.py --limit 15 --no-rerank
        """),
    )
    # Tìm file CSV mặc định: ưu tiên test/ subfolder, fallback về root
    _default_csv = PROJECT_ROOT / "test" / "test_retrieval_50q.csv"
    if not _default_csv.exists():
        _default_csv = PROJECT_ROOT / "test_retrieval_50q.csv"
    parser.add_argument(
        "--csv",
        default=str(_default_csv),
        help="Đường dẫn tới file CSV test (default: tự tìm test_retrieval_50q.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Số lượng jobs lấy từ hybrid_search (default: 10, min: 10)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        default=False,
        help="Bỏ qua cross-encoder rerank (nhanh hơn, để test RRF thuần)",
    )
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"[ERROR] Không tìm thấy file CSV: {args.csv}")
        sys.exit(1)

    evaluator = RetrievalEvaluator(
        csv_path  = args.csv,
        limit     = args.limit,
        no_rerank = args.no_rerank,
    )
    evaluator.run()


if __name__ == "__main__":
    main()