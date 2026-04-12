"""
generate_human_eval.py — Tự động gen câu trả lời cho Human Eval
================================================================
Đọc từng query trong CSV → gọi đúng pipeline của hệ thống (flask_serve)
→ lưu response vào cột mới trong cùng file.

Yêu cầu: Flask server phải đang chạy trước khi chạy script này.
  python flask_serve.py   (terminal khác)

Cách chạy:
  cd career_bot_v6
  python generate_human_eval.py
  python generate_human_eval.py --input test/human_eval_queries.csv
  python generate_human_eval.py --input test/human_eval_queries.csv --delay 1.5
  python generate_human_eval.py --resume   # tiếp tục nếu bị ngắt giữa chừng

Cột thêm vào CSV:
  bot_response   — câu trả lời từ bot
  bot_route      — route thực tế bot phân loại (job_search / career_advice / chitchat)
  bot_confidence — confidence score của router
  latency_s      — thời gian phản hồi (giây)
  status         — ok | error | timeout | skipped
  generated_at   — timestamp
"""

import argparse
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
API_ENDPOINT = "http://127.0.0.1:5001/api/v1/chat"
API_TIMEOUT  = 120  # giây

# Các cột sẽ được thêm vào CSV
OUTPUT_COLS = ["bot_response", "bot_route", "bot_confidence", "latency_s", "status", "generated_at"]


# ── Kiểm tra server ───────────────────────────────────────────────────────────
def _check_server() -> bool:
    try:
        r = requests.get("http://127.0.0.1:5001/api/v1/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


# ── Gọi API ───────────────────────────────────────────────────────────────────
def _call_api(query: str, session_id: str) -> dict:
    """
    Gọi đúng endpoint /api/v1/chat của flask_serve.py.
    Trả về dict với các key: content, route, confidence, latency_s, status.
    """
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            API_ENDPOINT,
            json={"query": query, "session_id": session_id},
            timeout=API_TIMEOUT,
        )
        latency = time.perf_counter() - t0

        if resp.status_code == 200:
            data = resp.json()
            return {
                "content":    data.get("content", ""),
                "route":      data.get("route", ""),
                "confidence": data.get("confidence", 0.0),
                "latency_s":  round(latency, 3),
                "status":     "ok",
            }
        else:
            return {
                "content":    f"[API ERROR {resp.status_code}] {resp.text[:300]}",
                "route":      "",
                "confidence": 0.0,
                "latency_s":  round(latency, 3),
                "status":     f"error_{resp.status_code}",
            }

    except requests.exceptions.Timeout:
        return {
            "content":    f"[TIMEOUT after {API_TIMEOUT}s]",
            "route":      "",
            "confidence": 0.0,
            "latency_s":  round(time.perf_counter() - t0, 3),
            "status":     "timeout",
        }
    except requests.exceptions.ConnectionError:
        return {
            "content":    "[CONNECTION ERROR] Server không phản hồi.",
            "route":      "",
            "confidence": 0.0,
            "latency_s":  round(time.perf_counter() - t0, 3),
            "status":     "connection_error",
        }
    except Exception as e:
        return {
            "content":    f"[ERROR] {e}",
            "route":      "",
            "confidence": 0.0,
            "latency_s":  round(time.perf_counter() - t0, 3),
            "status":     "error",
        }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Gen câu trả lời cho Human Eval CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python generate_human_eval.py
  python generate_human_eval.py --input test/human_eval_queries.csv
  python generate_human_eval.py --delay 2.0
  python generate_human_eval.py --resume
  python generate_human_eval.py --dry-run   # test 3 query đầu không lưu
        """,
    )
    parser.add_argument(
        "--input", default="test/human_eval_queries.csv",
        help="Đường dẫn file CSV (default: test/human_eval_queries.csv)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay giữa các request (giây, default: 1.0) — tránh rate limit Groq",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Bỏ qua các query đã có bot_response, chỉ chạy các query còn trống",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chạy thử 3 query đầu, không ghi vào file",
    )
    parser.add_argument(
        "--api", default=API_ENDPOINT,
        help=f"API endpoint (default: {API_ENDPOINT})",
    )
    args = parser.parse_args()

    # Resolve path (tìm từ project root nếu cần)
    csv_path = Path(args.input)
    if not csv_path.exists():
        # Thử tìm relative từ file script
        alt = Path(__file__).parent / args.input
        if alt.exists():
            csv_path = alt
        else:
            print(f"[ERROR] Không tìm thấy file: {args.input}")
            sys.exit(1)

    # ── Kiểm tra server ───────────────────────────────────────────────────────
    print(f"\n[Check] Kiểm tra Flask server tại {args.api}...", end=" ", flush=True)
    if not _check_server():
        print("FAILED")
        print("\n[ERROR] Flask server chưa chạy hoặc không phản hồi.")
        print("  → Mở terminal khác và chạy: python flask_serve.py")
        print("  → Sau đó chạy lại script này.\n")
        sys.exit(1)
    print("OK ✓")

    # ── Đọc CSV ───────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, dtype=str)
    df = df.fillna("")
    print(f"[Load] {len(df)} queries từ {csv_path}")

    if "query" not in df.columns:
        print("[ERROR] CSV thiếu cột 'query'.")
        sys.exit(1)

    # Thêm cột output nếu chưa có
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""

    # ── Xác định queries cần chạy ────────────────────────────────────────────
    if args.resume:
        pending_mask = df["bot_response"].eq("") | df["status"].isin(["", "timeout", "error", "connection_error"])
        pending_idx  = df[pending_mask].index.tolist()
        skipped      = len(df) - len(pending_idx)
        if skipped > 0:
            print(f"[Resume] Bỏ qua {skipped} queries đã có response, chạy {len(pending_idx)} queries còn lại.")
    else:
        pending_idx = df.index.tolist()

    if args.dry_run:
        pending_idx = pending_idx[:3]
        print(f"[Dry-run] Chỉ chạy {len(pending_idx)} queries đầu, KHÔNG lưu file.")

    if not pending_idx:
        print("[Done] Tất cả queries đã có response. Dùng --resume=False để chạy lại.")
        sys.exit(0)

    # ── Mỗi session_id riêng per query (để không bị context leak giữa queries) ──
    # Đây là điểm quan trọng: human eval cần mỗi query độc lập
    print(f"\n[Run] Bắt đầu gen {len(pending_idx)} queries (delay={args.delay}s giữa mỗi request)...\n")
    print(f"{'ID':>4}  {'Route':<15} {'Status':<12} {'Latency':>8}  Query")
    print("-" * 80)

    ok_count = err_count = 0

    for pos, idx in enumerate(pending_idx, 1):
        row   = df.loc[idx]
        query = str(row["query"]).strip()
        qid   = str(row.get("id", idx))

        if not query:
            df.at[idx, "status"] = "skipped"
            df.at[idx, "generated_at"] = datetime.now().isoformat()
            print(f"{qid:>4}  {'—':<15} {'skipped':<12} {'—':>8}  (trống)")
            continue

        # Session ID độc lập mỗi query → không bị ảnh hưởng context câu trước
        session_id = f"human_eval_{uuid.uuid4().hex[:8]}"

        result = _call_api(query, session_id)

        # Ghi vào dataframe
        df.at[idx, "bot_response"]   = result["content"]
        df.at[idx, "bot_route"]      = result["route"]
        df.at[idx, "bot_confidence"] = result["confidence"]
        df.at[idx, "latency_s"]      = result["latency_s"]
        df.at[idx, "status"]         = result["status"]
        df.at[idx, "generated_at"]   = datetime.now().isoformat()

        status_str = result["status"]
        if result["status"] == "ok":
            ok_count += 1
        else:
            err_count += 1

        query_preview = query[:45] + "…" if len(query) > 45 else query
        print(
            f"{qid:>4}  {result['route']:<15} {status_str:<12} "
            f"{result['latency_s']:>7.2f}s  {query_preview}"
        )

        # Lưu ngay vào file sau mỗi query (an toàn nếu bị crash giữa chừng)
        if not args.dry_run:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Delay giữa requests — quan trọng để không bị rate limit Groq
        if pos < len(pending_idx) and args.delay > 0:
            time.sleep(args.delay)

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  HOÀN THÀNH")
    print(f"  Tổng      : {len(pending_idx)} queries")
    print(f"  Thành công: {ok_count}")
    print(f"  Lỗi       : {err_count}")

    if not args.dry_run:
        print(f"  File lưu  : {csv_path.resolve()}")
        # In preview 3 dòng đầu có response
        sample = df[df["status"] == "ok"].head(3)
        if len(sample):
            print(f"\n  Preview (3 dòng đầu):")
            for _, r in sample.iterrows():
                preview = str(r["bot_response"])[:80].replace("\n", " ")
                print(f"    [{r.get('id','')}] {preview}…")
    else:
        print("  (Dry-run: không ghi file)")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()