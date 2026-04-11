"""
eval_qwen3_reranker.py — Đánh giá Recall@1/5/10 với Qwen3-Reranker-0.6B
========================================================================
Tái sử dụng candidate cache từ eval_jina_reranker.py (không fetch lại Qdrant).

Tối ưu tốc độ:
  [OPT-1] Candidate cache lưu/đọc từ disk → không load EmbeddingModel + Qdrant
           nếu đã chạy eval_jina_reranker.py trước
  [OPT-2] Gom TẤT CẢ pairs (50 queries x 30 candidates = 1500 pairs) vào
           1 danh sách phẳng, chạy inference theo batch_size=64 → giảm overhead
           Python loop tối đa
  [OPT-3] Qwen3-0.6B nhỏ (~1.2GB fp16), nhanh hơn nhiều so với 8B

Cách chạy:
  python eval_qwen3_reranker.py                     # dùng lại cache từ jina
  python eval_qwen3_reranker.py --no-rerank         # baseline RRF
  python eval_qwen3_reranker.py --reset-candidates  # fetch lại Qdrant
  python eval_qwen3_reranker.py --batch-size 32     # giảm nếu OOM
"""
from __future__ import annotations

import math, os, re, sys, time, types, pickle, argparse
from pathlib import Path
from datetime import datetime

_here = Path(__file__).resolve().parent
PROJECT_ROOT = _here
for _c in [_here, _here.parent, _here.parent.parent]:
    if (_c / "embedding_model").is_dir() and (_c / "rag").is_dir():
        PROJECT_ROOT = _c; break
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
import pandas as pd

# ── Query Parser (copy từ flask_serve.py) ─────────────────────────────────────
_LOC_KW = {
    "hà nội":"ha_noi","hanoi":"ha_noi","hn":"ha_noi",
    "hồ chí minh":"ho_chi_minh","hcm":"ho_chi_minh",
    "tp.hcm":"ho_chi_minh","sài gòn":"ho_chi_minh",
    "cần thơ":"can_tho","hải phòng":"hai_phong",
    "bình dương":"binh_duong","đồng nai":"dong_nai",
    "remote":"remote","work from home":"remote","wfh":"remote",
}
_SAL_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:[-\u2013đến]\s*(\d+(?:[.,]\d+)?))?\s*(triệu|tr|million|m)",
    re.IGNORECASE,
)
_FRESHER_KW = ["fresher","mới ra trường","chưa có kinh nghiệm","sinh viên","intern",
               "mới học","tự học","chưa biết gì","học việc","thực tập"]
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
        filters["salary_min"] = round(min(lo, hi), 1)
    if any(k in q for k in _FRESHER_KW):  filters["experience_norm"] = "fresher"
    elif any(k in q for k in _JUNIOR_KW): filters["experience_norm"] = "junior"
    elif any(k in q for k in _SENIOR_KW): filters["experience_norm"] = "senior"
    if any(k in q for k in _LINK_KW):     filters["link_only"] = True
    kw = query
    for loc in _LOC_KW: kw = kw.replace(loc, "")
    kw = _SAL_RE.sub("", kw); kw = re.sub(r"\s+", " ", kw).strip()
    return kw, filters

# ── Metrics ───────────────────────────────────────────────────────────────────
KS = [1, 5, 10]

def recall_at_k(ret, exp, k):
    if not exp: return 0.0
    return sum(1 for r in ret[:k] if r in exp) / len(exp)

def hit_at_k(ret, exp, k):
    return any(r in exp for r in ret[:k])

# ── Doc text ──────────────────────────────────────────────────────────────────
def _clean(v, fb=""):
    if not v or str(v).strip() in ("N/A","nan","None",""): return fb
    return str(v).strip()

def _doc_text(job):
    co = f"Công ty: {_clean(job.get('company'))}. " if _clean(job.get("company")) else ""
    return (
        f"Vị trí: {_clean(job.get('title'),'?')}. {co}"
        f"Lương: {_clean(job.get('salary_raw'),'Thỏa thuận')}. "
        f"Địa điểm: {_clean(job.get('location'))}. "
        f"Cấp bậc: {_clean(job.get('level'))}. "
        f"Kinh nghiệm: {_clean(job.get('experience'),'Không yêu cầu')}. "
        f"Yêu cầu: {(job.get('yeu_cau') or '')[:500]}"
    )

# ── Candidate cache (dùng chung với eval_jina_reranker.py) ────────────────────
def _cand_cache_path(csv_path: Path) -> Path:
    return csv_path.parent / "candidates_cache.pkl"

def _load_cand_cache(csv_path: Path) -> dict:
    p = _cand_cache_path(csv_path)
    if p.exists():
        with open(p, "rb") as f: data = pickle.load(f)
        print(f"[Cache] Loaded {len(data)} queries từ {p.name}")
        return data
    return {}

def _save_cand_cache(csv_path: Path, cache: dict):
    p = _cand_cache_path(csv_path)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f: pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)

# ── Qwen3-Reranker-0.6B ───────────────────────────────────────────────────────
# Không dùng env var vì .env đang set sai (0.5B không tồn tại)
QWEN_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"

_SYS    = ("Judge whether the Document meets the requirements based on the Query "
           "and the Instruct provided below. "
           "Note only output a single word 'yes' or 'no' — no other text.")
_INST   = ("Tìm kiếm công việc phù hợp với yêu cầu của ứng viên. "
           "Đánh giá xem Document (mô tả công việc) có đáp ứng Query (yêu cầu tìm việc) không.")
_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

def _prompt(query: str, doc: str) -> str:
    return (
        f"<|im_start|>system\n{_SYS}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"<Instruct>: {_INST}\n<Query>: {query}\n<Document>: {doc}\n"
        f"<|im_end|>\n{_SUFFIX}"
    )

def load_qwen3():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # embedding_model/core.py set HF_HUB_OFFLINE=1 ở module level →
    # unset trước khi load Qwen3 để cho phép download từ HF Hub
    os.environ.pop("HF_HUB_OFFLINE",       None)
    os.environ.pop("TRANSFORMERS_OFFLINE",  None)

    print(f"[Qwen3] Loading {QWEN_MODEL_ID}...", end=" ", flush=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(
        QWEN_MODEL_ID, trust_remote_code=True, padding_side="left",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID, trust_remote_code=True, device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.eval()

    yes_id = tok.encode("yes", add_special_tokens=False)[-1]
    no_id  = tok.encode("no",  add_special_tokens=False)[-1]
    device = next(model.parameters()).device
    print(f"OK ({time.time()-t0:.1f}s) | device={device} | yes={yes_id} no={no_id}")
    return model, tok, yes_id, no_id


def score_all_flat(
    queries_jobs: list[tuple[str, list[dict]]],
    model, tok, yes_id: int, no_id: int,
    batch_size: int = 64,
) -> list[list[float]]:
    """
    [OPT-2] Gom TẤT CẢ pairs từ mọi query vào 1 list phẳng, inference 1 lần.
    Tránh overhead khởi tạo vòng lặp per-query.
    """
    import torch

    # Build flat prompts
    flat: list[str] = []
    lengths: list[int] = []
    for kw, jobs in queries_jobs:
        chunk = [_prompt(kw, _doc_text(j)) for j in jobs]
        flat.extend(chunk)
        lengths.append(len(chunk))

    total = len(flat)
    print(f"[Qwen3] {total} pairs ({len(queries_jobs)} queries) | batch_size={batch_size}")
    t0 = time.time()

    flat_scores: list[float] = []
    for i in range(0, total, batch_size):
        chunk = flat[i : i + batch_size]
        inputs = tok(
            chunk, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model(**inputs).logits   # (B, seq_len, vocab)
        last = logits[:, -1, :]               # padding_side=left → đúng vị trí cuối
        for row in last:
            ey = math.exp(row[yes_id].item())
            en = math.exp(row[no_id].item())
            flat_scores.append(ey / (ey + en + 1e-9))
        print(f"  scored {min(i+batch_size, total)}/{total}...", end="\r")

    print(f"  scored {total}/{total} — {time.time()-t0:.1f}s total          ")

    # Tách lại theo query
    result: list[list[float]] = []
    idx = 0
    for n in lengths:
        result.append(flat_scores[idx : idx + n])
        idx += n
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    _def = PROJECT_ROOT / "test" / "test_retrieval_50q.csv"
    if not _def.exists(): _def = PROJECT_ROOT / "test_retrieval_50q.csv"
    parser.add_argument("--csv",              default=str(_def))
    parser.add_argument("--candidates",       type=int, default=30)
    parser.add_argument("--batch-size",       type=int, default=1,
                        help="Inference batch size (default: 64, giảm nếu OOM)")
    parser.add_argument("--no-rerank",        action="store_true", default=False)
    parser.add_argument("--reset-candidates", action="store_true", default=False,
                        help="Xoá candidate cache, fetch lại từ Qdrant")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    N = max(args.candidates, max(KS))

    print("\n" + "="*65)
    print("  Career Bot v6 — Retrieval Eval: Qwen3-Reranker-0.6B")
    print("="*65)
    print(f"  CSV        : {csv_path}")
    print(f"  Model      : {QWEN_MODEL_ID}")
    print(f"  Candidates : {N}")
    print(f"  Batch size : {args.batch_size}")
    print("="*65 + "\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    df    = pd.read_csv(csv_path)
    valid = {"job_search","job_search+filter","career_advice","career_advice+filter"}
    df    = df[df["type"].isin(valid)].reset_index(drop=True)
    rows  = df.to_dict("records")

    # ── Candidate cache ───────────────────────────────────────────────────────
    if args.reset_candidates:
        p = _cand_cache_path(csv_path)
        if p.exists(): p.unlink(); print("[Cache] Reset\n")

    cand_cache = _load_cand_cache(csv_path)
    todo       = [r for r in rows if r["id"] not in cand_cache]

    if todo:
        print(f"\n[Fetch] {len(todo)} queries chưa có candidates...\n")
        from embedding_model.core import EmbeddingModel
        from rag.core import RAG
        t0 = time.time(); embed = EmbeddingModel()
        print(f"[Init] EmbeddingModel ({time.time()-t0:.1f}s)")
        t0 = time.time(); rag = RAG(embedding_model=embed)
        print(f"[Init] RAG ({time.time()-t0:.1f}s)\n")

        orig = rag._rerank
        def _bypass(sr, q, jobs, top_n=10):
            return sorted(jobs, key=lambda j: j.get("_rrf_rank", 999))[:N]
        rag._rerank = types.MethodType(_bypass, rag)

        kws_todo = [_parse_query(r["query"])[0] or r["query"] for r in todo]
        fil_todo = [_parse_query(r["query"])[1] for r in todo]
        print(f"[Embed] Batch embed {len(todo)} queries...", end=" ", flush=True)
        t0 = time.time()
        vecs = embed.get_query_embeddings_batch(kws_todo)
        print(f"done ({time.time()-t0:.1f}s)\n")

        try:
            for i, (row, kw, qvec, filters) in enumerate(
                zip(todo, kws_todo, vecs, fil_todo), 1
            ):
                jobs = rag.hybrid_search(query=kw, query_vec=qvec, filters=filters, limit=N)
                cand_cache[row["id"]] = {"kw": kw, "jobs": jobs}
                _save_cand_cache(csv_path, cand_cache)
                print(f"  [{i:>3}/{len(todo)}] {row['id']} → {len(jobs)} candidates  ✓")
        finally:
            rag._rerank = orig
        print()
    else:
        print(f"[Cache] Tất cả {len(rows)} queries đã có → bỏ qua fetch\n")

    # ── Score tất cả pairs 1 lần ──────────────────────────────────────────────
    all_scores = None
    if not args.no_rerank:
        model, tok, yes_id, no_id = load_qwen3()
        print()
        queries_jobs = [(cand_cache[r["id"]]["kw"], cand_cache[r["id"]]["jobs"]) for r in rows]
        all_scores   = score_all_flat(queries_jobs, model, tok, yes_id, no_id, args.batch_size)
        print()

    # ── Tính metrics ──────────────────────────────────────────────────────────
    print("-"*65)
    records = []
    for i, row in enumerate(rows, 1):
        qid      = row["id"]
        expected = set(e.strip() for e in str(row["expected_job_ids"]).split("|") if e.strip())
        jobs     = cand_cache[qid]["jobs"]
        kw       = cand_cache[qid]["kw"]

        if args.no_rerank or all_scores is None:
            ranked = sorted(jobs, key=lambda j: j.get("_rrf_rank", 999))[:10]
        else:
            ranked = [j for j, _ in sorted(
                zip(jobs, all_scores[i-1]), key=lambda x: x[1], reverse=True
            )][:10]

        retrieved = [j["job_id"] for j in ranked if j.get("job_id")]
        rec  = {k: recall_at_k(retrieved, expected, k) for k in KS}
        hits = {k: hit_at_k(retrieved, expected, k)    for k in KS}

        records.append({
            "id":qid, "query":row["query"], "keyword":kw,
            "difficulty":row.get("difficulty","?"), "type":row.get("type","?"),
            "expected":expected, "retrieved":retrieved,
            **{f"recall@{k}":rec[k]  for k in KS},
            **{f"hit@{k}":   hits[k] for k in KS},
        })

        hs = " | ".join(f"H@{k}={'✓' if hits[k] else '✗'}" for k in KS)
        rs = " | ".join(f"R@{k}={rec[k]:.2f}" for k in KS)
        print(f"  [{i:>3}/{len(rows)}] {qid} [{row.get('difficulty','?'):>6}] {hs} | {rs}")
        if not hits[1] and expected:
            print(f"           ↳ expected : {expected}")
            print(f"           ↳ got top3 : {[j.get('title','?') for j in ranked[:3]]}")

    # ── Report ────────────────────────────────────────────────────────────────
    lbl = "no_rerank" if args.no_rerank else "qwen3_0.6b"
    rdf = pd.DataFrame(records)

    print("\n" + "="*65)
    print(f"  SUMMARY — {len(rows)} queries | mode={lbl}")
    print("="*65)

    print(f"\n{'Metric':<15} {'@1':>8} {'@5':>8} {'@10':>8}"); print("-"*40)
    for m in ("recall","hit"):
        vals = [rdf[f"{m}@{k}"].mean() for k in KS]
        print(f"{m.capitalize()+'@K':<15}" + "".join(f"{v:>8.3f}" for v in vals))

    print(f"\n{'Difficulty':<12} {'N':>4} {'@1':>8} {'@5':>8} {'@10':>8}"); print("-"*40)
    for d in ["easy","medium","hard"]:
        sub = rdf[rdf["difficulty"]==d]
        if not len(sub): continue
        vals = [sub[f"recall@{k}"].mean() for k in KS]
        print(f"{d:<12} {len(sub):>4}" + "".join(f"{v:>8.3f}" for v in vals))

    print(f"\n{'Type':<24} {'N':>4} {'@1':>8} {'@5':>8} {'@10':>8}"); print("-"*52)
    for t in sorted(rdf["type"].unique()):
        sub  = rdf[rdf["type"]==t]
        vals = [sub[f"recall@{k}"].mean() for k in KS]
        print(f"{t:<24} {len(sub):>4}" + "".join(f"{v:>8.3f}" for v in vals))

    missed = rdf[rdf["hit@10"]==False]
    if len(missed):
        print(f"\n▶ Missed @10 ({len(missed)})")
        for _, r in missed.iterrows():
            print(f"  [{r['id']}] {r['query'][:60]}")
            print(f"    expected : {r['expected']}")
            print(f"    retrieved: {r['retrieved'][:5]}")

    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = csv_path.parent / f"eval_qwen3_{lbl}_{ts}.csv"
    rdf[["id","query","keyword","difficulty","type",
         "recall@1","recall@5","recall@10",
         "hit@1","hit@5","hit@10","expected","retrieved",
    ]].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n✅ Kết quả lưu tại: {out}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()