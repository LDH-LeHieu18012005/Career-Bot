"""
generate_human_eval.py — Tạo dữ liệu cho human evaluation
============================================================
Hỗ trợ cả Jina v3 và Qwen3 Cross-Encoder reranker.
Tự động đọc cấu hình từ .env.

Cách chạy:
  python generate_human_eval.py                    # dùng reranker trong .env
  python generate_human_eval.py --reranker jina_v3
  python generate_human_eval.py --reranker qwen3
  python generate_human_eval.py --limit 20 --delay 1.5
  python generate_human_eval.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env trước khi parse arguments
load_dotenv()

HERE = Path(__file__).resolve().parent


# ── CSV reader (safe, RFC-4180 compliant) ─────────────────────────────────────
def _read_csv(path: str) -> list[dict]:
    """Đọc CSV bằng csv.DictReader — xử lý đúng quoted fields có dấu phẩy."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _load_existing(out_path: str) -> dict[str, dict]:
    """Load existing items từ output JSON để resume."""
    p = Path(out_path)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return {item["id"]: item for item in data.get("items", [])}
    except Exception as e:
        print(f"[Warn] Không đọc được checkpoint ({e}) → bắt đầu mới")
        return {}


def _save(out_path: str, items: list[dict], reranker: str = "", reranker_model: str = ""):
    """Lưu kết quả ra JSON (ghi đè toàn bộ, gọi sau mỗi request)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "total":          len(items),
            "generated_at":   datetime.now().isoformat(),
            "reranker":       reranker,
            "reranker_model": reranker_model,
            "device":         os.getenv("QWEN3_RERANKER_DEVICE") or "auto",
        },
        "items": items,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ── Health check ──────────────────────────────────────────────────────────────
def _check_server(api_base: str) -> bool:
    try:
        import urllib.request, urllib.error
        req = urllib.request.Request(
            f"{api_base}/api/v1/health",
            headers={"User-Agent": "generate_human_eval/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── Call chat API ─────────────────────────────────────────────────────────────
def _call_chat(api_base: str, query: str, session_id: str, timeout: int = 90) -> dict:
    """Gọi POST /api/v1/chat bằng urllib (không phụ thuộc requests)."""
    import urllib.request, urllib.error
    import json as _json

    payload = _json.dumps({
        "query":      query,
        "session_id": session_id,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{api_base}/api/v1/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return _json.loads(body)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tạo dữ liệu human eval từ API CareerBot (hỗ trợ Jina v3 & Qwen3 reranker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ví dụ:
  python generate_human_eval.py
  python generate_human_eval.py --reranker qwen3
  python generate_human_eval.py --limit 20 --delay 0.8
  python generate_human_eval.py --resume --out test/my_eval.json""",
    )
    parser.add_argument("--api",     default="http://127.0.0.1:5001",
                        help="Base URL của Flask server")
    parser.add_argument("--csv",     default=str(HERE / "test" / "test_retrieval_50q.csv"),
                        help="Path tới CSV test queries")
    parser.add_argument("--out",     default=None,
                        help="Path lưu output JSON (nếu không chỉ định sẽ tự sinh theo reranker)")
    parser.add_argument("--reranker", default=None,
                        help="Tên reranker: jina_v3 hoặc qwen3 (ưu tiên command line > .env)")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Giới hạn số queries (0 = tất cả)")
    parser.add_argument("--delay",   type=float, default=1.0,
                        help="Giây chờ giữa các request")
    parser.add_argument("--resume",  action="store_true",
                        help="Bỏ qua queries đã có trong output file")
    parser.add_argument("--timeout", type=int, default=90,
                        help="Timeout mỗi request (giây)")

    args = parser.parse_args()

    # ── Xử lý reranker từ .env và command line ───────────────────────────────
    env_backend = os.getenv("RERANKER_BACKEND", "jina_v3").strip().lower()

    if args.reranker is None:
        args.reranker = env_backend
    else:
        args.reranker = args.reranker.strip().lower()

    # Xác định model name để ghi vào meta
    if args.reranker == "qwen3" or args.reranker.startswith("qwen"):
        reranker_model = os.getenv("QWEN3_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
        reranker_display = "qwen3"
    else:
        reranker_model = "jina-reranker-v3"
        reranker_display = "jina_v3"

    # Tự động sinh tên file output nếu không chỉ định
    if args.out is None:
        args.out = str(HERE / "test" / f"human_eval_{reranker_display}.json")

    print(f"[Reranker] {reranker_display.upper()}  |  Model: {reranker_model}")

    # ── Kiểm tra server ──────────────────────────────────────────────────────
    print(f"\n[Check] Kiểm tra Flask server tại {args.api}/api/v1/health ...", end=" ", flush=True)
    if _check_server(args.api):
        print("OK ✓")
    else:
        print("FAILED ✗")
        print(f"\n  ⚠  Không kết nối được server tại {args.api}")
        print("     Hãy chắc chắn Flask đang chạy: python flask_serve.py")
        sys.exit(1)

    # ── Đọc CSV ──────────────────────────────────────────────────────────────
    if not Path(args.csv).exists():
        print(f"\n[ERROR] Không tìm thấy: {args.csv}")
        sys.exit(1)

    try:
        rows = _read_csv(args.csv)
    except Exception as e:
        print(f"\n[ERROR] Không đọc được CSV: {e}")
        sys.exit(1)

    if args.limit > 0:
        rows = rows[:args.limit]

    print(f"[CSV]   {len(rows)} queries từ {args.csv}")

    # ── Resume ───────────────────────────────────────────────────────────────
    existing: dict[str, dict] = {}
    if args.resume:
        existing = _load_existing(args.out)
        if existing:
            print(f"[Resume] {len(existing)} queries đã có → bỏ qua")

    # ── Chuẩn bị ─────────────────────────────────────────────────────────────
    items: list[dict] = list(existing.values())
    pending = [r for r in rows if r["id"] not in existing]
    errors = 0

    print(f"[Run]   Cần gọi API cho {len(pending)} queries\n")
    print("─" * 75)

    # ── Gọi API ──────────────────────────────────────────────────────────────
    for i, row in enumerate(pending, 1):
        qid   = row.get("id", "?")
        query = row.get("query", "")
        qtype = row.get("type", "?")
        diff  = row.get("difficulty", "?")
        exp_t = row.get("expected_titles", "")

        label = f"[{i:>3}/{len(pending)}] {qid} [{diff:>6}]  {query[:55]}"
        print(label, end=" … ", flush=True)
        t0 = time.perf_counter()

        session_id = str(uuid.uuid4())

        item: dict = {
            "id":              qid,
            "query":           query,
            "type":            qtype,
            "difficulty":      diff,
            "expected_titles": exp_t,
            "generated_at":    datetime.now().isoformat(),
        }

        try:
            data = _call_chat(args.api, query, session_id, timeout=args.timeout)
            elapsed = time.perf_counter() - t0
            item.update({
                "response":   data.get("content", ""),
                "route":      data.get("route", ""),
                "confidence": data.get("confidence", 0),
                "elapsed_s":  round(elapsed, 3),
            })
            print(f"OK  ({elapsed:.1f}s)  route={item['route']}")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            item.update({
                "response":   f"[ERROR] {e}",
                "route":      "error",
                "confidence": 0,
                "elapsed_s":  round(elapsed, 3),
            })
            print(f"FAIL ({elapsed:.1f}s)  {e}")
            errors += 1

        items.append(item)
        _save(args.out, items, reranker=reranker_display, reranker_model=reranker_model)

        if args.delay > 0 and i < len(pending):
            time.sleep(args.delay)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("─" * 75)
    ok_count = len(items) - errors
    print(f"\n✅ Hoàn thành: {len(items)} items  ({ok_count} OK, {errors} lỗi)")
    print(f"   Output: {args.out}")
    print(f"\n▶  Bước tiếp theo: python human_eval_server.py\n")


if __name__ == "__main__":
    main()