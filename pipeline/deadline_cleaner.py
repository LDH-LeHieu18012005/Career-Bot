"""
pipeline/deadline_cleaner.py — Tự Động Xóa Tin Tuyển Dụng Hết Hạn
===================================================================
Xóa các điểm (points) trong Qdrant có `deadline_ts` < thời điểm hiện tại.

Chạy thủ công:
    python pipeline/deadline_cleaner.py

Chạy với báo cáo chi tiết (dry-run, không xóa thật):
    python pipeline/deadline_cleaner.py --dry-run

Chạy scheduled (mỗi N giờ):
    python pipeline/deadline_cleaner.py --schedule --interval 24

Chạy 1 lần lúc cụ thể mỗi ngày (dùng --cron-time):
    python pipeline/deadline_cleaner.py --schedule --cron-time 02:00

Tích hợp trong Docker:
    Xem docker-compose.yml — service `deadline_cleaner`

Cơ chế hoạt động:
    1. Kết nối Qdrant
    2. Dùng Qdrant filter: deadline_ts IS NOT NULL AND deadline_ts < now_ts
    3. Lấy danh sách job_id bị ảnh hưởng (để log)
    4. Gọi client.delete() với cùng filter
    5. Ghi log kết quả

Field cần có trong payload:
    deadline_ts  : float  — Unix timestamp (thêm từ chunker.py v2)
    deadline     : str    — Chuỗi gốc "dd/MM/yyyy" (để log cho dễ đọc)
    job_id       : str    — ID job (để log)
    title        : str    — Tên vị trí (để log)
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DeadlineCleaner")

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL  = os.getenv("QDRANT_URL", "")
QDRANT_KEY  = os.getenv("QDRANT_API_KEY", "")
COLLECTION  = os.getenv("QDRANT_COLLECTION", "topcv_jobs_v3")

# Số điểm tối đa lấy để preview (dry-run / log)
PREVIEW_LIMIT = int(os.getenv("CLEANER_PREVIEW_LIMIT", "100"))

# ── Qdrant Helper ─────────────────────────────────────────────────────────────

def get_qdrant_client():
    from qdrant_client import QdrantClient
    if not QDRANT_URL or not QDRANT_KEY:
        raise ValueError(
            "Thiếu QDRANT_URL hoặc QDRANT_API_KEY trong .env\n"
            "Tạo cluster miễn phí: https://cloud.qdrant.io"
        )
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=60)


def build_expired_filter(now_ts: float):
    """
    Tạo Qdrant filter: deadline_ts tồn tại VÀ deadline_ts < now_ts
    """
    from qdrant_client.http.models import (
        Filter, FieldCondition, Range, IsNullCondition, PayloadField
    )

    return Filter(
        must=[
            # deadline_ts PHẢI tồn tại (không null)
            FieldCondition(
                key="deadline_ts",
                range=Range(gt=0, lt=now_ts),   # 0 < deadline_ts < now_ts
            ),
        ]
    )


# ── Preview (dry-run) ─────────────────────────────────────────────────────────

def preview_expired(client, expired_filter, limit: int = PREVIEW_LIMIT) -> list:
    """
    Lấy danh sách các điểm sẽ bị xóa (không xóa thật).
    Returns: list of dicts với job_id, title, deadline
    """
    results = client.scroll(
        collection_name = COLLECTION,
        scroll_filter   = expired_filter,
        limit           = limit,
        with_payload    = ["job_id", "title", "deadline", "deadline_ts", "location"],
        with_vectors    = False,
    )
    points = results[0]  # (points, next_page_offset)

    expired = []
    seen_jobs = set()
    for p in points:
        pay = p.payload or {}
        jid = pay.get("job_id", str(p.id))
        if jid in seen_jobs:
            continue
        seen_jobs.add(jid)
        expired.append({
            "job_id":   jid,
            "title":    pay.get("title", "N/A"),
            "deadline": pay.get("deadline", "N/A"),
            "location": pay.get("location", "N/A"),
        })

    return expired


# ── Delete Expired ────────────────────────────────────────────────────────────

def delete_expired(
    dry_run:          bool = False,
    now_ts:           Optional[float] = None,
) -> dict:
    """
    Xóa tất cả points hết hạn khỏi Qdrant.

    Returns:
        dict với keys: deleted_count, deleted_jobs, dry_run, timestamp
    """
    if now_ts is None:
        now_ts = datetime.now(timezone.utc).timestamp()

    now_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    log.info(f"[Cleaner] ⏰ Thời điểm kiểm tra: {now_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    client = get_qdrant_client()

    # Kiểm tra collection tồn tại
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        log.warning(f"[Cleaner] Collection '{COLLECTION}' chưa tồn tại. Bỏ qua.")
        return {"deleted_count": 0, "deleted_jobs": [], "dry_run": dry_run}

    expired_filter = build_expired_filter(now_ts)

    # ── Preview những job sẽ bị xóa ─────────────────────────────────────────
    expired_jobs = preview_expired(client, expired_filter, limit=PREVIEW_LIMIT)

    if not expired_jobs:
        log.info("[Cleaner] ✅ Không có tin tuyển dụng nào hết hạn. Tất cả còn hiệu lực.")
        return {"deleted_count": 0, "deleted_jobs": [], "dry_run": dry_run}

    log.info(f"[Cleaner] 📋 Tìm thấy {len(expired_jobs)} job đã hết hạn (preview {PREVIEW_LIMIT}):")
    for j in expired_jobs[:10]:  # In tối đa 10 jobs
        log.info(f"   • [{j['deadline']}] {j['title']} — {j['location']} ({j['job_id']})")
    if len(expired_jobs) > 10:
        log.info(f"   ... và {len(expired_jobs) - 10} job khác")

    if dry_run:
        log.info("[Cleaner] 🔍 DRY-RUN: Không xóa thật. Dùng không có --dry-run để xóa.")
        return {
            "deleted_count": len(expired_jobs),
            "deleted_jobs":  expired_jobs,
            "dry_run":       True,
        }

    # ── Xóa thật sự từ Qdrant ────────────────────────────────────────────────
    log.info(f"[Cleaner] 🗑️  Đang xóa tất cả points hết hạn từ collection '{COLLECTION}'...")

    delete_result = client.delete(
        collection_name = COLLECTION,
        points_selector = expired_filter,
    )

    log.info(f"[Cleaner] ✅ Qdrant xóa thành công. Status: {delete_result.status}")

    # Tổng số points còn lại
    info = client.get_collection(COLLECTION)
    log.info(f"[Cleaner] 📊 Collection còn lại: {info.points_count:,} points")

    # Ghi log vào file (để audit)
    _write_audit_log(expired_jobs, now_ts, dry_run=False)

    return {
        "deleted_count": len(expired_jobs),
        "deleted_jobs":  expired_jobs,
        "dry_run":       False,
        "timestamp":     now_dt.isoformat(),
    }


# ── Audit Log ─────────────────────────────────────────────────────────────────

def _write_audit_log(deleted_jobs: list, now_ts: float, dry_run: bool) -> None:
    """Ghi audit log ra file JSON Lines."""
    log_dir  = ROOT / "data" / "cleaner_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"cleanup_{datetime.now().strftime('%Y%m')}.jsonl"

    entry = {
        "timestamp":     datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
        "dry_run":       dry_run,
        "deleted_count": len(deleted_jobs),
        "deleted_jobs":  deleted_jobs,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log.info(f"[AuditLog] Ghi log tại: {log_file}")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_scheduler(interval_hours: int = 24, cron_time: Optional[str] = None, dry_run: bool = False) -> None:
    """
    Chạy cleaner theo lịch.

    Args:
        interval_hours: Chạy mỗi N giờ (dùng khi không có cron_time)
        cron_time:      Chạy lúc HH:MM mỗi ngày, VD: "02:00"
        dry_run:        Chỉ preview, không xóa thật
    """
    import schedule

    mode_str = f"lúc {cron_time} hàng ngày" if cron_time else f"mỗi {interval_hours} giờ"
    log.info(f"[Scheduler] ⏱️  Lịch chạy: {mode_str}")
    log.info(f"[Scheduler]    Dry-run: {dry_run}")
    log.info("[Scheduler]    Ctrl+C để dừng\n")

    def job():
        log.info("[Scheduler] 🔔 Bắt đầu chu kỳ dọn dẹp...")
        try:
            result = delete_expired(dry_run=dry_run)
            log.info(
                f"[Scheduler] Chu kỳ hoàn tất. "
                f"Đã xóa: {result['deleted_count']} jobs "
                f"({'dry-run' if result['dry_run'] else 'thật'})"
            )
        except Exception as e:
            log.error(f"[Scheduler] ❌ Lỗi trong chu kỳ dọn dẹp: {e}", exc_info=True)

    # Lên lịch
    if cron_time:
        schedule.every().day.at(cron_time).do(job)
    else:
        schedule.every(interval_hours).hours.do(job)

    # Chạy ngay 1 lần khi khởi động
    log.info("[Scheduler] Chạy lần đầu ngay khi khởi động...")
    job()

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Kiểm tra mỗi phút
    except KeyboardInterrupt:
        log.info("[Scheduler] Dừng scheduler.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tự động xóa tin tuyển dụng hết hạn khỏi Qdrant"
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ liệt kê jobs hết hạn, KHÔNG xóa thật"
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Chạy ở chế độ scheduled (lặp liên tục)"
    )
    parser.add_argument(
        "--interval", type=int, default=24,
        help="Khoảng cách giữa các lần chạy (giờ). Mặc định: 24"
    )
    parser.add_argument(
        "--cron-time",
        help="Giờ chạy hàng ngày theo dạng HH:MM. VD: --cron-time 02:00"
    )
    parser.add_argument(
        "--simulate-date",
        help="Test với ngày giả định (dd/MM/yyyy). VD: --simulate-date 01/05/2026"
    )

    args = parser.parse_args()

    # Xử lý simulate-date để test
    now_ts = None
    if args.simulate_date:
        from datetime import datetime, timezone
        try:
            sim_dt  = datetime.strptime(args.simulate_date, "%d/%m/%Y")
            now_ts  = sim_dt.replace(tzinfo=timezone.utc).timestamp()
            log.info(f"[Simulate] Giả lập ngày: {args.simulate_date}")
        except ValueError:
            log.error("--simulate-date phải dạng dd/MM/yyyy. VD: 01/05/2026")
            sys.exit(1)

    if args.schedule:
        try:
            import schedule as _
        except ImportError:
            log.error("Cài thư viện: pip install schedule")
            sys.exit(1)

        run_scheduler(
            interval_hours = args.interval,
            cron_time      = args.cron_time,
            dry_run        = args.dry_run,
        )
    else:
        # Chạy 1 lần
        result = delete_expired(
            dry_run        = args.dry_run,
            now_ts         = now_ts,
        )

        print(f"\n{'='*50}")
        print(f"  {'[DRY-RUN] ' if result['dry_run'] else ''}KẾT QUẢ DỌN DẸP")
        print(f"{'='*50}")
        print(f"  Jobs hết hạn tìm thấy : {result['deleted_count']}")
        print(f"  Đã xóa thật           : {'Không (dry-run)' if result['dry_run'] else 'Có'}")
        if result["deleted_jobs"]:
            print(f"\n  Danh sách ({min(5, len(result['deleted_jobs']))} đầu):")
            for j in result["deleted_jobs"][:5]:
                print(f"    • [{j['deadline']}] {j['title']}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
