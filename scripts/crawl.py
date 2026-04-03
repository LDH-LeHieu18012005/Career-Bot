"""
scripts/crawl.py — TopCV Crawler v6
=====================================
Crawl dữ liệu việc làm từ TopCV → data/TopCV_Jobs_Data.jsonl

Cải tiến so với v4:
  [V6-1] Hỗ trợ crawl nhiều category (không chỉ IT)
  [V6-2] Exponential backoff khi gặp lỗi liên tiếp
  [V6-3] Progress report mỗi 50 jobs
  [V6-4] Thêm field 'level' và 'company_size'
  [V6-5] Resume từ file đã có (không crawl lại URL cũ)

Chạy:
    python scripts/crawl.py
    python scripts/crawl.py --target 3000 --category it
    python scripts/crawl.py --target 5000 --category all
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime

# Thêm parent vào path để import utils nếu cần
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
except ImportError:
    print("❌ Cài selenium trước: pip install selenium")
    sys.exit(1)


# ── Config ─────────────────────────────────────────────────────────────────────

OUTPUT_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "TopCV_Jobs_Data.jsonl")
MAX_NO_NEW_PAGES = 5

# Danh sách category URL TopCV
CATEGORIES = {
    "it":        "https://www.topcv.vn/tim-viec-lam-cong-nghe-thong-tin-cr257",
    "data":      "https://www.topcv.vn/tim-viec-lam-khoa-hoc-du-lieu-cr258",
    "design":    "https://www.topcv.vn/tim-viec-lam-thiet-ke-cr26",
    "marketing": "https://www.topcv.vn/tim-viec-lam-marketing-cr27",
    "finance":   "https://www.topcv.vn/tim-viec-lam-tai-chinh-ke-toan-cr22",
    "all":       "https://www.topcv.vn/viec-lam",
}


# ── Utils ─────────────────────────────────────────────────────────────────────

def random_sleep(a: float = 1.0, b: float = 2.5):
    time.sleep(random.uniform(a, b))


def clean_text(text: str) -> str:
    if not text:
        return "N/A"
    return re.sub(r"\s+", " ", text).strip()


def load_existing_urls(output_file: str) -> set:
    urls = set()
    if not os.path.exists(output_file):
        return urls
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                url  = data.get("metadata", {}).get("url", "")
                if url:
                    urls.add(url)
            except json.JSONDecodeError:
                pass
    return urls


def save_jsonl(data: dict, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


# ── Driver ─────────────────────────────────────────────────────────────────────

def create_driver(headless: bool = False) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()

    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1366,768")

    # Tắt load ảnh để tăng tốc
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# ── Anti-bot ─────────────────────────────────────────────────────────────────

def human_scroll(driver):
    """Scroll giả lập người dùng để trigger lazy load."""
    for _ in range(3):
        scroll_y = random.randint(400, 800)
        driver.execute_script(f"window.scrollBy(0, {scroll_y});")
        time.sleep(random.uniform(0.3, 0.7))
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(random.uniform(0.5, 1.0))


def get_text_safe(driver, xpath: str, timeout: int = 3, default: str = "N/A") -> str:
    """Lấy text từ xpath, trả default nếu không tìm thấy."""
    try:
        el = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        text = el.text.strip()
        return text if text else default
    except (TimeoutException, NoSuchElementException):
        return default


def get_text_any(driver, xpaths: list[str], timeout: int = 3, default: str = "N/A") -> str:
    """Thử nhiều xpath, trả kết quả đầu tiên tìm được."""
    for xp in xpaths:
        result = get_text_safe(driver, xp, timeout=timeout)
        if result != "N/A":
            return result
    return default


# ── Job Link Extraction ───────────────────────────────────────────────────────

JOB_LINK_XPATHS = [
    "//div[contains(@class,'box-job-list')]//h3//a",
    "//div[contains(@class,'job-item')]//h3//a",
    "//div[contains(@class,'job-list-item')]//a[contains(@class,'job-title')]",
    "//a[contains(@href,'/viec-lam/') and contains(@class,'title')]",
]


def get_job_links(driver) -> list[str]:
    links = []
    for xp in JOB_LINK_XPATHS:
        try:
            els = driver.find_elements(By.XPATH, xp)
            for el in els:
                href = el.get_attribute("href")
                if href and "viec-lam" in href and href not in links:
                    links.append(href)
        except Exception:
            pass
    return links


# ── Job Detail Extraction ─────────────────────────────────────────────────────

def extract_job_detail(driver, url: str) -> dict | None:
    """Crawl chi tiết 1 job. Trả None nếu lỗi hoặc thiếu dữ liệu."""
    try:
        driver.get(url)
        time.sleep(random.uniform(1.2, 2.0))

        title = get_text_any(driver, [
            "//h1[contains(@class,'title')]",
            "//h1",
        ])

        company = get_text_any(driver, [
            "//a[contains(@class,'company-name')]",
            "//h2//a",
            "//div[contains(@class,'company')]//h2",
        ])

        salary = get_text_any(driver, [
            "//div[contains(@class,'salary')]//span[contains(@class,'salary-text')]",
            "//i[contains(@class,'salary')]/parent::div//span",
            "//div[contains(text(),'Mức lương')]/following-sibling::div",
            "//label[contains(text(),'Mức lương')]/following-sibling::span",
        ])

        location = get_text_any(driver, [
            "//div[contains(@class,'location')]//span",
            "//i[contains(@class,'location')]/parent::div",
            "//div[contains(text(),'Địa điểm')]/following-sibling::div",
        ])

        experience = get_text_any(driver, [
            "//div[contains(text(),'Kinh nghiệm')]/parent::div//div[contains(@class,'value')]",
            "//div[contains(text(),'Kinh nghiệm')]/following-sibling::div",
            "//label[contains(text(),'Kinh nghiệm')]/following-sibling::span",
        ])

        level = get_text_any(driver, [
            "//div[contains(text(),'Cấp bậc')]/following-sibling::div",
            "//div[contains(text(),'Cấp bậc')]/parent::div//div[contains(@class,'value')]",
            "//label[contains(text(),'Cấp bậc')]/following-sibling::span",
        ])

        deadline = get_text_any(driver, [
            "//div[contains(@class,'deadline')]",
            "//div[contains(@class,'actions-label')]",
            "//span[contains(text(),'Hạn nộp')]",
        ])

        # Content sections
        description = get_text_any(driver, [
            "//div[contains(@class,'job-description')]//div[contains(text(),'Mô tả')]/following-sibling::div",
            "//*[contains(@class,'section')][contains(text(),'Mô tả')]/following-sibling::div",
            "//*[contains(text(),'Mô tả công việc')]/following-sibling::*",
        ])

        requirements = get_text_any(driver, [
            "//*[contains(text(),'Yêu cầu ứng viên')]/following-sibling::div",
            "//*[contains(text(),'Yêu cầu')]/following-sibling::div",
        ])

        benefits = get_text_any(driver, [
            "//*[contains(text(),'Quyền lợi')]/following-sibling::div",
            "//*[contains(text(),'Phúc lợi')]/following-sibling::div",
        ])

        if title == "N/A":
            return None

        return {
            "metadata": {
                "url":        url,
                "title":      clean_text(title),
                "company":    clean_text(company),
                "salary":     clean_text(salary),
                "location":   clean_text(location),
                "experience": clean_text(experience),
                "level":      clean_text(level),
                "deadline":   clean_text(deadline),
                "source":     "TopCV",
                "crawl_time": datetime.now().isoformat(),
            },
            "content": {
                "description":  clean_text(description),
                "requirements": clean_text(requirements),
                "benefits":     clean_text(benefits),
            },
        }

    except Exception as e:
        print(f"   ⚠️  Lỗi extract: {str(e)[:60]}")
        return None


# ── Page Processor ────────────────────────────────────────────────────────────

def process_page(
    page: int,
    category_url: str,
    driver,
    existing_urls: set,
    output_file: str,
) -> int:
    """Crawl 1 page danh sách việc làm. Trả về số job mới."""
    url = f"{category_url}?page={page}"
    try:
        driver.get(url)
        time.sleep(random.uniform(1.5, 2.5))
        human_scroll(driver)

        job_urls = get_job_links(driver)

        # Retry nếu không lấy được links
        if not job_urls:
            print(f"   🔄 Retry page {page}...")
            time.sleep(3)
            driver.refresh()
            time.sleep(2)
            human_scroll(driver)
            job_urls = get_job_links(driver)

        new_count = 0
        for job_url in job_urls:
            if job_url in existing_urls:
                continue

            job_data = extract_job_detail(driver, job_url)
            if job_data:
                save_jsonl(job_data, output_file)
                existing_urls.add(job_url)
                new_count += 1
                print(f"   ✅ {job_data['metadata']['title'][:45]}")

            random_sleep(1.0, 2.0)

        return new_count

    except Exception as e:
        print(f"❌ Lỗi page {page}: {e}")
        return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TopCV Crawler v6")
    parser.add_argument("--target",   type=int, default=5000, help="Số job cần crawl")
    parser.add_argument("--category", type=str, default="it",
                        choices=list(CATEGORIES.keys()), help="Category cần crawl")
    parser.add_argument("--headless", action="store_true", help="Chạy ẩn (no GUI)")
    parser.add_argument("--output",   type=str, default=OUTPUT_FILE, help="Output file path")
    args = parser.parse_args()

    category_url = CATEGORIES[args.category]
    output_file  = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"🚀 Career Bot v6 Crawler")
    print(f"   Category : {args.category} ({category_url})")
    print(f"   Target   : {args.target:,} jobs")
    print(f"   Output   : {output_file}")
    print(f"   Headless : {args.headless}")
    print()

    existing_urls = load_existing_urls(output_file)
    total         = len(existing_urls)
    print(f"🔁 Đã có sẵn: {total:,} jobs\n")

    driver    = create_driver(headless=args.headless)
    page      = 1
    no_new    = 0
    errors    = 0
    MAX_ERR   = 10

    try:
        while True:
            print(f"\n🔹 Page {page}  |  Total: {total:,}/{args.target:,}")

            new_jobs = process_page(page, category_url, driver, existing_urls, output_file)
            total   += new_jobs

            if new_jobs == 0:
                no_new += 1
                errors += 1
                wait    = min(30, 5 * no_new)
                print(f"   ⚠️  Không có job mới ({no_new}/{MAX_NO_NEW_PAGES}). Chờ {wait}s...")
                time.sleep(wait)
            else:
                no_new = 0
                errors = 0

            # Progress report
            if total % 50 < new_jobs:
                print(f"\n📊 Progress: {total:,}/{args.target:,} jobs ({total/args.target*100:.1f}%)")

            if total >= args.target:
                print(f"\n🎯 Đủ {args.target:,} jobs → DONE")
                break

            if no_new >= MAX_NO_NEW_PAGES:
                print(f"\n🛑 {MAX_NO_NEW_PAGES} pages liên tiếp không có job mới → DONE")
                break

            if errors >= MAX_ERR:
                print(f"\n🛑 Quá nhiều lỗi liên tiếp ({MAX_ERR}) → DONE")
                break

            page += 1
            time.sleep(random.uniform(3, 6))

    except KeyboardInterrupt:
        print("\n⏹️  Dừng bởi user (Ctrl+C)")
    finally:
        driver.quit()
        print(f"\n✅ HOÀN THÀNH: {total:,} jobs → {output_file}")


if __name__ == "__main__":
    main()
