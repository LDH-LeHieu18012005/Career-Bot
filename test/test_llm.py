"""
eval_bleu_rouge.py — Đánh giá chất lượng Retrieval bằng BLEU & ROUGE
======================================================================
So sánh:
  - Hypothesis : tiêu đề các job được hệ thống retrieve (từ eval_results)
  - Reference  : tiêu đề các job đúng (expected_titles trong test_retrieval_50q)

Metric:
  - BLEU-1, BLEU-2, BLEU-4  (sacrebleu corpus-level)
  - ROUGE-1, ROUGE-2, ROUGE-L (rouge_score)

Cách chạy:
    pip install sacrebleu rouge-score pandas
    python eval_bleu_rouge.py

Output:
    bleu_rouge_results.csv   — kết quả từng query
    bleu_rouge_summary.txt   — tổng hợp theo difficulty / type
"""

import ast
import re
import json
import os
import csv
from collections import defaultdict
from datetime import datetime

import pandas as pd

# ── Cài thư viện nếu thiếu ───────────────────────────────────────────────────
try:
    import sacrebleu
except ImportError:
    os.system("pip install sacrebleu -q")
    import sacrebleu

try:
    from rouge_score import rouge_scorer
except ImportError:
    os.system("pip install rouge-score -q")
    from rouge_score import rouge_scorer


# ══════════════════════════════════════════════════════════════════════════════
# ── CẤU HÌNH ĐƯỜNG DẪN (chỉnh tại đây) ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR       = r"C:\Users\DELL\Desktop\career_bot_v6"

EVAL_FILE      = os.path.join(BASE_DIR, "test", "eval_results_20260409_2330.csv")
TEST_FILE      = os.path.join(BASE_DIR, "test", "test_retrieval_50q.csv")
JSONL_FILE     = os.path.join(BASE_DIR, "data", "TopCV_Jobs_Data.jsonl")

OUT_CSV        = os.path.join(BASE_DIR, "test", "bleu_rouge_results.csv")
OUT_SUMMARY    = os.path.join(BASE_DIR, "test", "bleu_rouge_summary.txt")

# Số lượng job retrieve dùng để tính (top-k)
TOP_K = 6

# ══════════════════════════════════════════════════════════════════════════════


def build_job_title_map(jsonl_path: str) -> dict[str, str]:
    """Đọc JSONL → {job_id: title}."""
    mapping = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row  = json.loads(line)
                meta = row.get("metadata", {}) or {}
                url  = meta.get("url", "")
                title = meta.get("title", "")
                m = re.search(r"/(\d{5,})\.html", url.split("?")[0])
                if m and title:
                    mapping["job_" + m.group(1)] = title
            except Exception:
                continue
    return mapping


def normalize(text: str) -> str:
    """Lowercase + chuẩn hoá khoảng trắng — dùng cho BLEU/ROUGE tiếng Việt."""
    return re.sub(r"\s+", " ", text.lower().strip())


def ids_to_titles(job_ids: list[str], title_map: dict[str, str]) -> list[str]:
    """Chuyển danh sách job_id → danh sách title (bỏ qua id không tìm được)."""
    return [title_map[jid] for jid in job_ids if jid in title_map]


def compute_rouge(hypothesis: str, reference: str) -> dict[str, float]:
    """Tính ROUGE-1, ROUGE-2, ROUGE-L (F1)."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(hypothesis: str, reference: str) -> dict[str, float]:
    """Tính BLEU-1, BLEU-2, BLEU-3, BLEU-4 ở cấp độ sentence."""
    hyp_tok = hypothesis.split()
    ref_tok = reference.split()

    if not hyp_tok or not ref_tok:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    def ngram_precision(hyp, ref, n):
        from collections import Counter
        hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(max(0, len(hyp)-n+1)))
        ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(max(0, len(ref)-n+1)))
        matches = sum((hyp_ngrams & ref_ngrams).values())
        total   = sum(hyp_ngrams.values())
        return matches / total if total > 0 else 0.0

    import math
    bp = min(1.0, math.exp(1 - len(ref_tok) / len(hyp_tok))) if hyp_tok else 0.0

    p1 = ngram_precision(hyp_tok, ref_tok, 1)
    p2 = ngram_precision(hyp_tok, ref_tok, 2)
    p3 = ngram_precision(hyp_tok, ref_tok, 3)
    p4 = ngram_precision(hyp_tok, ref_tok, 4)

    # BLEU-N = BP × geometric mean của p1..pN
    def geo_mean(precs):
        precs = [max(p, 1e-9) for p in precs]
        return math.exp(sum(math.log(p) for p in precs) / len(precs))

    bleu1 = bp * p1                                     if p1 > 0 else 0.0
    bleu2 = bp * geo_mean([p1, p2])                     if p1 > 0 and p2 > 0 else 0.0
    bleu3 = bp * geo_mean([p1, p2, p3])                 if p1 > 0 and p2 > 0 and p3 > 0 else 0.0
    bleu4 = bp * geo_mean([p1, p2, p3, p4])             if p1 > 0 and p2 > 0 and p3 > 0 and p4 > 0 else 0.0

    return {
        "bleu1": round(bleu1, 4),
        "bleu2": round(bleu2, 4),
        "bleu3": round(bleu3, 4),
        "bleu4": round(bleu4, 4),
    }


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BLEU & ROUGE Evaluation — Career Bot Retrieval")
    print("=" * 60)

    # 1. Load dữ liệu
    print("\n[1/4] Đọc dữ liệu...")
    df_eval = pd.read_csv(EVAL_FILE, encoding="utf-8-sig")
    df_test = pd.read_csv(TEST_FILE, encoding="utf-8-sig")

    # Merge theo id để lấy expected_titles
    df = df_eval.merge(
        df_test[["id", "expected_titles", "eval_metric"]],
        on="id", how="left"
    )
    print(f"  Queries : {len(df)}")

    # 2. Build job_id → title map
    print("[2/4] Xây dựng bảng job_id → title từ JSONL...")
    title_map = build_job_title_map(JSONL_FILE)
    print(f"  Tìm thấy {len(title_map):,} job titles")

    # 3. Tính BLEU & ROUGE từng query
    print("[3/4] Tính BLEU & ROUGE từng query...")
    records = []

    for _, row in df.iterrows():
        qid        = row["id"]
        query      = row["query"]
        difficulty = row.get("difficulty", "")
        qtype      = row.get("type", "")
        eval_metric = row.get("eval_metric", "")

        # ── Parse retrieved job_ids ──
        try:
            retrieved_ids = ast.literal_eval(row["retrieved"])[:TOP_K]
        except Exception:
            retrieved_ids = []

        # ── Hypothesis: ghép title các job được retrieve ──
        ret_titles  = ids_to_titles(retrieved_ids, title_map)
        hypothesis  = normalize(" | ".join(ret_titles)) if ret_titles else ""

        # ── Reference: expected_titles từ test file ──
        exp_raw     = str(row.get("expected_titles", "") or "")
        ref_titles  = [t.strip() for t in exp_raw.split("|") if t.strip()]
        reference   = normalize(" | ".join(ref_titles)) if ref_titles else ""

        # ── Tính metric ──
        if hypothesis and reference:
            bleu  = compute_bleu(hypothesis, reference)
            rouge = compute_rouge(hypothesis, reference)
        else:
            bleu  = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
            rouge = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}

        record = {
            "id":               qid,
            "query":            query,
            "difficulty":       difficulty,
            "type":             qtype,
            "eval_metric":      eval_metric,
            "expected_titles":  " | ".join(ref_titles),
            "retrieved_titles": " | ".join(ret_titles),
            **bleu,
            **rouge,
        }
        records.append(record)

        print(
            f"  {qid}  B1={bleu['bleu1']:.3f}  B2={bleu['bleu2']:.3f}"
            f"  B3={bleu['bleu3']:.3f}  B4={bleu['bleu4']:.3f}"
            f"  R1={rouge['rouge1_f']:.3f}  RL={rouge['rougeL_f']:.3f}"
            f"  [{difficulty}/{qtype}]"
        )

    # 4. Lưu CSV kết quả từng query
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[4/4] Đã lưu: {OUT_CSV}")

    # ── Tổng hợp ──────────────────────────────────────────────────────────────
    metric_cols = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge1_f", "rouge2_f", "rougeL_f"]

    def summarize(label: str, sub_df: pd.DataFrame) -> str:
        n = len(sub_df)
        if n == 0:
            return ""
        parts = [f"  {label} (n={n})"]
        for col in metric_cols:
            parts.append(f"    {col:<12}: {sub_df[col].mean():.4f}")
        return "\n".join(parts)

    lines = [
        "=" * 60,
        "  BLEU & ROUGE Summary",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Top-K retrieved: {TOP_K}",
        "=" * 60,
        "",
        "── OVERALL ──────────────────────────────────────────────",
        summarize("ALL", df_out),
        "",
        "── BY DIFFICULTY ────────────────────────────────────────",
    ]
    for diff in ["easy", "medium", "hard"]:
        sub = df_out[df_out["difficulty"] == diff]
        lines.append(summarize(diff.upper(), sub))

    lines += ["", "── BY TYPE ──────────────────────────────────────────────"]
    for qtype in sorted(df_out["type"].dropna().unique()):
        sub = df_out[df_out["type"] == qtype]
        lines.append(summarize(qtype, sub))

    lines += ["", "── BY EVAL_METRIC ───────────────────────────────────────"]
    for em in sorted(df_out["eval_metric"].dropna().unique()):
        sub = df_out[df_out["eval_metric"] == em]
        lines.append(summarize(em, sub))

    summary = "\n".join(l for l in lines if l is not None)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + summary)
    print(f"\nĐã lưu tổng hợp: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()