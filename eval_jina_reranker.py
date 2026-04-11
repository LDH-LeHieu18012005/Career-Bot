"""
eval_jina_reranker.py — Đánh giá Recall@1/5/10 với Jina Reranker v3
=====================================================================
Pipeline:
  Phase 1  Pre-fetch — mỗi query gọi hybrid_search() DUY NHẤT 1 lần,
           lưu vào candidate_cache.
  Phase 2  Rerank — Jina v3 (Qwen3 backbone, causal LM scoring) → top-10.
  Phase 3  Báo cáo Recall/Hit @1/5/10 và lưu CSV.

Model: jinaai/jina-reranker-v3
Architecture: Qwen3 causal LM backbone (KHÔNG phải SequenceClassification)

CHANGELOG:
  [FIX-1] Dùng AutoModelForCausalLM thay vì AutoModelForSequenceClassification.
          jina-reranker-v3 phiên bản Qwen3 KHÔNG có classification head —
          score.weight trong Qwen3ForSequenceClassification là randomly
          initialized (garbage).  Scoring đúng = P("yes") ở last token.

  [FIX-2] model.config.pad_token_id = tokenizer.pad_token_id
          Qwen3.forward() kiểm tra config.pad_token_id (KHÔNG phải
          tokenizer.pad_token).  Thiếu dòng này → crash khi batch > 1
          dù tokenizer.pad_token đã được set.
            ValueError: "Cannot handle batch sizes > 1 if no padding
            token is defined."  (modeling_qwen3.py:671)

  [FIX-3] _SAL_RE double backslash: )?\\s* → )?\s*

Cách chạy:
  python eval_jina_reranker.py
  python eval_jina_reranker.py --no-rerank
  python eval_jina_reranker.py --batch 2    # giảm nếu OOM
"""

from __future__ import annotations

import math, os, re, sys, time, types, argparse, textwrap
from pathlib import Path
from datetime import datetime

_here = Path(__file__).resolve().parent
PROJECT_ROOT = _here
for _c in [_here, _here.parent, _here.parent.parent]:
    if (_c / "embedding_model").is_dir() and (_c / "rag").is_dir():
        PROJECT_ROOT = _c
        break
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
import pandas as pd

# ── Query Parser (copy từ flask_serve.py) ─────────────────────────────────────
_LOC_KW = {
    "hà nội": "ha_noi", "hanoi": "ha_noi", "hn": "ha_noi",
    "hồ chí minh": "ho_chi_minh", "hcm": "ho_chi_minh",
    "tp.hcm": "ho_chi_minh", "sài gòn": "ho_chi_minh",
    "cần thơ": "can_tho", "hải phòng": "hai_phong",
    "bình dương": "binh_duong", "đồng nai": "dong_nai",
    "remote": "remote", "work from home": "remote", "wfh": "remote",
}
# [FIX-3] single backslash \s* (double backslash = literal '\' in input)
_SAL_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:[-\u2013\u0111\u1ebfn]\s*(\d+(?:[.,]\d+)?))?\s*(tri\u1ec7u|tr|million|m)",
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

def hit_at_k(retrieved, expected, k):
    return any(r in expected for r in retrieved[:k])

# ── Doc text builder ──────────────────────────────────────────────────────────
def _clean(v, fallback=""):
    if not v or str(v).strip() in ("N/A","nan","None",""): return fallback
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

# ── Jina Reranker v3 (Qwen3 causal LM) ───────────────────────────────────────
JINA_MODEL_ID = os.getenv("JINA_RERANKER_MODEL", "jinaai/jina-reranker-v3")

_SYS = (
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided below. "
    "Note only output a single word 'yes' or 'no' — no other text."
)
_INST = (
    "Tìm kiếm công việc phù hợp với yêu cầu của ứng viên. "
    "Đánh giá xem Document (mô tả công việc) có đáp ứng Query (yêu cầu tìm việc) không."
)
_SFXS = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

def _prompt(query, doc):
    return (
        f"<|im_start|>system\n{_SYS}\n<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {_INST}\n<Query>: {query}\n<Document>: {doc}\n<|im_end|>\n"
        f"{_SFXS}"
    )

def load_jina_reranker():
    """
    [FIX-1] Dùng AutoModelForCausalLM — jina-reranker-v3 (Qwen3 version)
             score bằng P(yes/no) tại last token, không có classification head.
    [FIX-2] model.config.pad_token_id = tokenizer.pad_token_id
             Qwen3.forward() checks config, không check tokenizer attr.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[Jina] Loading {JINA_MODEL_ID} (CausalLM)...", end=" ", flush=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(
        JINA_MODEL_ID, trust_remote_code=True, local_files_only=True,
        padding_side="left",   # causal LM: pad bên trái
    )
    if tok.pad_token is None:            # Qwen3 không có pad_token mặc định
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(   # [FIX-1]
        JINA_MODEL_ID, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.pad_token_id = tok.pad_token_id    # [FIX-2] propagate to config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    yes_id = tok.encode("yes", add_special_tokens=False)[-1]
    no_id  = tok.encode("no",  add_special_tokens=False)[-1]

    print(f"OK ({time.time()-t0:.1f}s) | device={device} | yes={yes_id} no={no_id}")
    return model, tok, device, yes_id, no_id

def _score_batch(query, docs, model, tok, yes_id, no_id):
    import torch
    prompts = [_prompt(query, d) for d in docs]
    inputs  = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=4096).to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**inputs).logits   # (B, seq, vocab)
    last = logits[:, -1, :]               # (B, vocab) — padding_side=left → correct pos
    scores = []
    for row in last:
        ey = math.exp(row[yes_id].item()); en = math.exp(row[no_id].item())
        scores.append(ey / (ey + en + 1e-9))
    return scores

def jina_rerank(query, jobs, model, tok, device, yes_id, no_id, top_n=10, batch=4):
    if not jobs: return []
    docs = [_doc_text(j) for j in jobs]; all_s = []
    for i in range(0, len(docs), batch):
        try:
            all_s.extend(_score_batch(query, docs[i:i+batch], model, tok, yes_id, no_id))
        except Exception as e:
            print(f"\n[Jina] batch {i//batch} error: {e}")
            all_s.extend([0.0] * len(docs[i:i+batch]))
    ranked = sorted(zip(jobs, all_s), key=lambda x: x[1], reverse=True)
    return [j for j,_ in ranked[:top_n]]

# ── Evaluator ─────────────────────────────────────────────────────────────────
class JinaEvaluator:
    def __init__(self, csv_path, candidates=30, no_rerank=False, batch=4):
        self.csv_path   = csv_path
        self.candidates = max(candidates, max(KS))
        self.no_rerank  = no_rerank
        self.batch      = batch
        self.results    = []
        self._banner(); self._load_pipeline(); self._load_dataset()

    def _banner(self):
        mode = "NO-RERANK (RRF)" if self.no_rerank else f"Jina v3 ({JINA_MODEL_ID}) | batch={self.batch}"
        print("\n" + "="*65)
        print("  Career Bot v6 — Retrieval Eval: Jina Reranker v3")
        print("="*65)
        print(f"  CSV        : {self.csv_path}")
        print(f"  Mode       : {mode}")
        print(f"  Candidates : {self.candidates}  (fetched ONCE per query)")
        print("="*65 + "\n")

    def _load_pipeline(self):
        from embedding_model.core import EmbeddingModel
        from rag.core import RAG
        print("[Init] Loading EmbeddingModel...")
        t0 = time.time(); self.embed = EmbeddingModel()
        print(f"[Init] EmbeddingModel ready ({time.time()-t0:.1f}s)\n")
        print("[Init] Connecting Qdrant + BM25...")
        t0 = time.time(); self.rag = RAG(embedding_model=self.embed)
        print(f"[Init] RAG ready ({time.time()-t0:.1f}s)\n")

    def _load_dataset(self):
        df = pd.read_csv(self.csv_path)
        valid = {"job_search","job_search+filter","career_advice","career_advice+filter"}
        self.df = df[df["type"].isin(valid)].reset_index(drop=True)
        print(f"[Data] {len(df)} total → {len(self.df)} retrieval queries\n")

    def _prefetch_all(self, rows):
        print(f"[Phase 1] Pre-fetching {self.candidates} candidates cho {len(rows)} queries...\n")
        cache = {}; orig = self.rag._rerank; N = self.candidates
        def _bypass(sr, q, jobs, top_n=10):
            return sorted(jobs, key=lambda j: j.get("_rrf_rank",999))[:N]
        self.rag._rerank = types.MethodType(_bypass, self.rag)
        try:
            for i, row in enumerate(rows, 1):
                kw, f = _parse_query(row["query"])
                qv    = self.embed.get_query_embedding(kw or row["query"])
                jobs  = self.rag.hybrid_search(query=kw or row["query"], query_vec=qv,
                                               filters=f, limit=N)
                cache[row["id"]] = jobs
                print(f"  [{i:>3}/{len(rows)}] {row['id']} → {len(jobs)} candidates")
        finally:
            self.rag._rerank = orig
        print(f"\n[Phase 1] Done\n"); return cache

    def run(self):
        rows  = self.df.to_dict("records")
        cache = self._prefetch_all(rows)

        jm = jt = jd = jy = jn = None
        if not self.no_rerank:
            print("[Phase 2] Loading Jina Reranker...\n")
            jm, jt, jd, jy, jn = load_jina_reranker(); print()

        lbl = "no_rerank" if self.no_rerank else "jina_v3"
        print(f"[Phase 2] Evaluating {len(rows)} queries (mode={lbl})...\n")
        print("-"*65)

        for i, row in enumerate(rows, 1):
            qid  = row["id"]; query = row["query"]
            kw,_ = _parse_query(query)
            exp  = set(e.strip() for e in str(row["expected_job_ids"]).split("|") if e.strip())
            diff = row.get("difficulty","?"); qt = row.get("type","?")
            cands = cache[qid]

            t0 = time.time()
            ranked = (sorted(cands, key=lambda j: j.get("_rrf_rank",999))[:10]
                      if self.no_rerank else
                      jina_rerank(kw or query, cands, jm, jt, jd, jy, jn,
                                  top_n=10, batch=self.batch))
            elapsed = time.time() - t0

            ret  = [j["job_id"] for j in ranked if j.get("job_id")]
            rec  = {k: recall_at_k(ret, exp, k) for k in KS}
            hits = {k: hit_at_k(ret, exp, k)    for k in KS}

            self.results.append({
                "id":qid,"query":query,"keyword":kw,"difficulty":diff,"type":qt,
                "expected":exp,"retrieved":ret,"t_rerank":elapsed,
                **{f"recall@{k}":rec[k] for k in KS},
                **{f"hit@{k}":hits[k]   for k in KS},
            })
            hs = " | ".join(f"H@{k}={'✓' if hits[k] else '✗'}" for k in KS)
            rs = " | ".join(f"R@{k}={rec[k]:.2f}" for k in KS)
            print(f"  [{i:>3}/{len(rows)}] {qid} [{diff:>6}] {hs} | {rs} | {elapsed:.2f}s")
            if not hits[1] and exp:
                print(f"           ↳ expected : {exp}")
                print(f"           ↳ got top3 : {[j.get('title','?') for j in ranked[:3]]}")

        self._report(lbl)

    def _report(self, lbl):
        df = pd.DataFrame(self.results); n = len(df)
        print("\n" + "="*65 + f"\n  SUMMARY — {n} queries | mode={lbl}\n" + "="*65)

        print(f"\n{'Metric':<15} {'@1':>8} {'@5':>8} {'@10':>8}"); print("-"*42)
        for m in ("recall","hit"):
            vals = [df[f"{m}@{k}"].mean() for k in KS]
            print(f"{m.capitalize()+'@K':<15}" + "".join(f"{v:>8.3f}" for v in vals))

        print(f"\n{'Difficulty':<12} {'N':>4} {'@1':>8} {'@5':>8} {'@10':>8}"); print("-"*42)
        for d in ["easy","medium","hard"]:
            sub = df[df["difficulty"]==d]
            if len(sub)==0: continue
            vals = [sub[f"recall@{k}"].mean() for k in KS]
            print(f"{d:<12} {len(sub):>4}" + "".join(f"{v:>8.3f}" for v in vals))

        print(f"\n{'Type':<24} {'N':>4} {'@1':>8} {'@5':>8} {'@10':>8}"); print("-"*52)
        for t in sorted(df["type"].unique()):
            sub = df[df["type"]==t]; vals = [sub[f"recall@{k}"].mean() for k in KS]
            print(f"{t:<24} {len(sub):>4}" + "".join(f"{v:>8.3f}" for v in vals))

        missed = df[df["hit@10"]==False]
        if len(missed):
            print(f"\n▶ Missed @10 ({len(missed)})")
            for _,r in missed.iterrows():
                print(f"  [{r['id']}] {r['query'][:60]}")
                print(f"    expected : {r['expected']}"); print(f"    retrieved: {r['retrieved'][:5]}")

        print(f"\nLatency — avg={df['t_rerank'].mean():.2f}s  max={df['t_rerank'].max():.2f}s")
        ts  = datetime.now().strftime("%Y%m%d_%H%M")
        out = Path(self.csv_path).parent / f"eval_jina_{lbl}_{ts}.csv"
        df[["id","query","keyword","difficulty","type",
            "recall@1","recall@5","recall@10","hit@1","hit@5","hit@10",
            "t_rerank","expected","retrieved"]].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n✅ Kết quả lưu tại: {out}\n" + "="*65 + "\n")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Eval Jina Reranker v3 — Recall@1/5/10")
    _def = PROJECT_ROOT / "test" / "test_retrieval_50q.csv"
    if not _def.exists(): _def = PROJECT_ROOT / "test_retrieval_50q.csv"
    parser.add_argument("--csv",        default=str(_def))
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--no-rerank",  action="store_true", default=False)
    parser.add_argument("--batch",      type=int, default=4, help="giảm nếu OOM")
    args = parser.parse_args()
    if not Path(args.csv).exists():
        print(f"[ERROR] Không tìm thấy: {args.csv}"); sys.exit(1)
    JinaEvaluator(args.csv, args.candidates, args.no_rerank, args.batch).run()

if __name__ == "__main__":
    main()