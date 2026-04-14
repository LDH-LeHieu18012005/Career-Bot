import os, sys, re, time, argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
os.chdir(_here)
load_dotenv(_here / ".env")

# ── Query Parser ──────────────────────────────────────────────────────────────
_LOC_KW = {
    "hà nội": "ha_noi", "hanoi": "ha_noi", "hn": "ha_noi",
    "hồ chí minh": "ho_chi_minh", "hcm": "ho_chi_minh",
    "tp.hcm": "ho_chi_minh", "sài gòn": "ho_chi_minh",
    "cần thơ": "can_tho", "hải phòng": "hai_phong",
    "bình dương": "binh_duong", "đồng nai": "dong_nai",
    "remote": "remote", "work from home": "remote", "wfh": "remote",
}
_SAL_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:[-\u2013\u0111\u1ebfn]\s*(\d+(?:[.,]\d+)?))?\s*(tri\u1ec7u|tr|million|m)",
    re.IGNORECASE,
)
_FRESHER_KW = ["fresher","mới ra trường","chưa có kinh nghiệm","sinh viên","intern","mới học","tự học","chưa biết gì","học việc","thực tập"]
_JUNIOR_KW  = ["junior","1 năm","2 năm","mới đi làm"]
_SENIOR_KW  = ["senior","lead","3 năm","4 năm","5 năm","nhiều năm"]
_LINK_KW    = ["link","url","ứng tuyển","apply","xin link"]

def _parse_query(query: str) -> tuple[str, dict]:
    q = query.lower(); filters: dict = {}
    for kw, norm in _LOC_KW.items():
        if kw in q: filters["location_norm"] = norm; break
    m = _SAL_RE.search(q)
    if m:
        lo = float(m.group(1).replace(",",".")); hi = float(m.group(2).replace(",",".")) if m.group(2) else lo
        if lo > 500: lo /= 1_000_000
        if hi > 500: hi /= 1_000_000
        filters["salary_min"] = round(min(lo,hi),1)
    if any(k in q for k in _FRESHER_KW): filters["experience_norm"] = "fresher"
    elif any(k in q for k in _JUNIOR_KW): filters["experience_norm"] = "junior"
    elif any(k in q for k in _SENIOR_KW): filters["experience_norm"] = "senior"
    if any(k in q for k in _LINK_KW): filters["link_only"] = True
    kw = query
    for loc in _LOC_KW: kw = kw.replace(loc, "")
    kw = _SAL_RE.sub("", kw); kw = re.sub(r"\s+", " ", kw).strip()
    return kw, filters

# ── Metrics ───────────────────────────────────────────────────────────────────
KS = [1, 5, 10]

def recall_at_k(retrieved, expected, k):
    if not expected: return 0.0
    return sum(1 for r in retrieved[:k] if r in expected) / len(expected)

# ── RRF Utility ───────────────────────────────────────────────────────────────
def _rrf(doc_lists: list[list[dict]], weights: list[float], c: int = 60) -> list[dict]:
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baselines")
    parser.add_argument("--csv", default="test/test_retrieval_50q.csv")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--mode", choices=["sparse", "dense", "hybrid", "all"], default="all",
                        help="Chọn baseline để đánh giá (mặc định: all)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] File {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    valid = {"job_search","job_search+filter","career_advice","career_advice+filter"}
    if "type" in df.columns:
        df = df[df["type"].isin(valid)].reset_index(drop=True)
    
    print(f"Loading {len(df)} queries from {args.csv}")
    print(f"Mode: {args.mode.upper()}")

    # Initialize RAG and Embedding
    from rag.core import RAG
    
    embed = None
    if args.mode in ["dense", "hybrid", "all"]:
        from embedding_model.core import EmbeddingModel
        print("\n[Init] Loading EmbeddingModel...")
        t0 = time.time()
        # Thêm biến môi trường để fix lỗi in emoji trên terminal Windows
        import sys
        if sys.platform == "win32":
            os.environ["PYTHONIOENCODING"] = "utf-8"
        embed = EmbeddingModel()
        print(f"[Init] EmbeddingModel ready ({time.time()-t0:.1f}s)")
    
    print("\n[Init] Connecting RAG...")
    t0 = time.time()
    rag = RAG(embedding_model=embed)
    print(f"[Init] RAG ready ({time.time()-t0:.1f}s)\n")

    results_sparse = []
    results_dense = []
    results_hybrid = []

    for i, row in df.iterrows():
        qid  = row.get("id", f"q{i}")
        query = row["query"]
        expected_str = str(row.get("expected_job_ids", ""))
        expected = set(e.strip() for e in expected_str.split("|") if e.strip())
        
        kw, filters = _parse_query(query)
        qdrant_filter = rag._build_filter(filters)
        
        # Deduplicate helper specifically matching the pipeline
        def _dedup(jobs_list):
            unique, seen_urls, seen_jids = [], set(), set()
            for j in jobs_list:
                jid = j.get("job_id", "")
                if not jid or jid in seen_jids: continue
                url = j.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    seen_jids.add(jid)
                    unique.append(j)
                elif not url:
                    seen_jids.add(jid)
                    unique.append(j)
            return unique

        # 1. Sparse Only (BM25)
        bm25_res = []
        ret_sparse = []
        if args.mode in ["sparse", "hybrid", "all"]:
            bm25_res = rag.bm25_search(kw or query, limit=args.top_k * 2, filters=filters)
            ret_sparse = [j["job_id"] for j in _dedup(bm25_res)]
        
        # 2. Dense Only (Qdrant + Jina)
        vec_res = []
        ret_dense = []
        if args.mode in ["dense", "hybrid", "all"]:
            qv = embed.get_query_embedding(kw or query)
            vec_res = rag.vector_search(qv, limit=args.top_k * 2, qdrant_filter=qdrant_filter)
            ret_dense = [j["job_id"] for j in _dedup(vec_res)]
        
        # 3. Dense + BM25 + RRF (k=60), no reranking
        ret_hybrid = []
        if args.mode in ["hybrid", "all"]:
            fused = _rrf([vec_res, bm25_res], weights=[0.6, 0.4], c=60)
            ret_hybrid = [j["job_id"] for j in _dedup(fused)]

        # Calculate metrics
        sparse_rec = {k: recall_at_k(ret_sparse, expected, k) for k in KS}
        dense_rec = {k: recall_at_k(ret_dense, expected, k) for k in KS}
        hybrid_rec = {k: recall_at_k(ret_hybrid, expected, k) for k in KS}

        results_sparse.append(sparse_rec)
        results_dense.append(dense_rec)
        results_hybrid.append(hybrid_rec)

        print(f"[{i+1}/{len(df)}] {qid}")
        if args.mode in ["sparse", "all"]: print(f"  Sparse: " + " | ".join(f"R@{k}={sparse_rec[k]:.2f}" for k in KS))
        if args.mode in ["dense",  "all"]: print(f"  Dense : " + " | ".join(f"R@{k}={dense_rec[k]:.2f}" for k in KS))
        if args.mode in ["hybrid", "all"]: print(f"  Hybrid: " + " | ".join(f"R@{k}={hybrid_rec[k]:.2f}" for k in KS))

    # Print Summary
    print("\n" + "="*50)
    print(" SUMMARY (Average Recall@K)")
    print("="*50)
    
    def print_summary(name, res_list):
        r1 = sum(r[1] for r in res_list) / len(res_list) if res_list else 0
        r5 = sum(r[5] for r in res_list) / len(res_list) if res_list else 0
        r10 = sum(r[10] for r in res_list) / len(res_list) if res_list else 0
        print(f"{name:<20} R@1: {r1:.3f} | R@5: {r5:.3f} | R@10: {r10:.3f}")

    if args.mode in ["sparse", "all"]: print_summary("Baseline 1: Sparse Only", results_sparse)
    if args.mode in ["dense",  "all"]: print_summary("Baseline 2: Dense Only", results_dense)
    if args.mode in ["hybrid", "all"]: print_summary("Baseline 3: Hybrid (RRF)", results_hybrid)
    print("="*50)

if __name__ == "__main__":
    main()
