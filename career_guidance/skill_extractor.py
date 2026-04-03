"""
career_guidance/skill_extractor.py — Skill Extraction & Taxonomy Builder v6
=============================================================================
Chiến lược:
  1. Dùng keyword matching với skill dictionary (~250 skills)
     → Không cần LLM per job → nhanh, không tốn token
  2. Scroll toàn bộ jobs từ Qdrant, match skills trong yeu_cau + mo_ta
  3. Build taxonomy: {skill: {count, experience_levels, avg_salary_min, jobs}}
  4. Cache ra data/skill_taxonomy.json (tự invalidate khi data thay đổi)

Dùng:
    extractor = SkillExtractor(qdrant_client, collection_name)
    taxonomy  = extractor.get_taxonomy()   # load cache hoặc build
    extractor.build()                      # force rebuild
"""

import os
import re
import json
import time
import unicodedata

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

_TAXONOMY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "skill_taxonomy.json")
)


# ── Skill Dictionary ──────────────────────────────────────────────────────────
# Cấu trúc: {canonical_name: [alias1, alias2, ...]}
SKILL_DICT: dict[str, list[str]] = {
    # ── Programming Languages ──
    "Python":         ["python"],
    "Java":           ["java", " java "],
    "JavaScript":     ["javascript", "js", "es6", "es2015"],
    "TypeScript":     ["typescript", "ts"],
    "C++":            ["c++", "cpp"],
    "C#":             ["c#", "csharp", ".net"],
    "PHP":            ["php"],
    "Go":             ["golang", " go "],
    "Rust":           ["rust"],
    "Ruby":           ["ruby", "rails", "ruby on rails"],
    "Swift":          ["swift"],
    "Kotlin":         ["kotlin"],
    "Dart":           ["dart"],
    "Scala":          ["scala"],
    "R":              [" r programming", "rlang"],

    # ── Web Frontend ──
    "React":          ["react", "reactjs", "react.js"],
    "Vue.js":         ["vue", "vuejs", "vue.js"],
    "Angular":        ["angular", "angularjs"],
    "Next.js":        ["next.js", "nextjs"],
    "Nuxt.js":        ["nuxt.js", "nuxtjs"],
    "HTML/CSS":       ["html", "css", "html5", "css3", "sass", "scss", "tailwind", "bootstrap"],
    "Redux":          ["redux", "redux toolkit", "zustand", "mobx"],
    "jQuery":         ["jquery"],
    "Webpack":        ["webpack", "vite", "babel", "rollup"],

    # ── Web Backend ──
    "Node.js":        ["node.js", "nodejs", "node", "express", "expressjs", "nestjs"],
    "Django":         ["django"],
    "Flask":          ["flask"],
    "FastAPI":        ["fastapi"],
    "Spring Boot":    ["spring", "spring boot", "spring mvc"],
    "Laravel":        ["laravel"],
    "ASP.NET":        ["asp.net", "aspnet", ".net core"],
    "GraphQL":        ["graphql"],
    "REST API":       ["rest api", "restful", "api design", "openapi", "swagger"],
    "gRPC":           ["grpc"],
    "Microservices":  ["microservice", "microservices", "kiến trúc vi dịch vụ"],

    # ── Mobile ──
    "Flutter":        ["flutter"],
    "React Native":   ["react native"],
    "Android":        ["android", "android sdk"],
    "iOS":            ["ios", "xcode", "objective-c"],

    # ── Database ──
    "SQL":            ["sql", "mysql", "mariadb", "oracle db"],
    "PostgreSQL":     ["postgresql", "postgres"],
    "MongoDB":        ["mongodb", "mongo"],
    "Redis":          ["redis"],
    "Elasticsearch":  ["elasticsearch", "elastic search", "opensearch"],
    "Firebase":       ["firebase"],
    "SQLite":         ["sqlite"],
    "Cassandra":      ["cassandra"],
    "ClickHouse":     ["clickhouse"],

    # ── Cloud & DevOps ──
    "AWS":            ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "GCP":            ["gcp", "google cloud", "bigquery"],
    "Azure":          ["azure", "microsoft azure"],
    "Docker":         ["docker", "dockerfile", "docker-compose"],
    "Kubernetes":     ["kubernetes", "k8s"],
    "CI/CD":          ["ci/cd", "cicd", "jenkins", "github actions", "gitlab ci", "circleci"],
    "Linux":          ["linux", "ubuntu", "centos", "bash", "shell script"],
    "Terraform":      ["terraform", "infrastructure as code", "iac"],
    "Nginx":          ["nginx", "apache"],
    "Kafka":          ["kafka", "apache kafka", "message queue", "rabbitmq"],

    # ── AI / ML / Data ──
    "Machine Learning":  ["machine learning", "ml", "học máy"],
    "Deep Learning":     ["deep learning", "neural network", "mạng nơ-ron"],
    "NLP":               ["nlp", "natural language processing", "xử lý ngôn ngữ"],
    "Computer Vision":   ["computer vision", "xử lý ảnh", "image processing"],
    "TensorFlow":        ["tensorflow", "tf"],
    "PyTorch":           ["pytorch", "torch"],
    "scikit-learn":      ["scikit-learn", "sklearn"],
    "Pandas":            ["pandas"],
    "NumPy":             ["numpy"],
    "Spark":             ["apache spark", "pyspark", "spark"],
    "Data Analysis":     ["data analysis", "phân tích dữ liệu", "data analytics", "eda"],
    "Power BI":          ["power bi", "powerbi"],
    "Tableau":           ["tableau"],
    "ETL":               ["etl", "data pipeline", "data warehouse", "dwh"],
    "LLM":               ["llm", "chatgpt", "openai", "langchain", "rag", "vector database"],
    "MLOps":             ["mlops", "model deployment", "serving model"],

    # ── Design & Product ──
    "Figma":          ["figma"],
    "Photoshop":      ["photoshop", "adobe photoshop"],
    "Illustrator":    ["illustrator", "adobe illustrator"],
    "Adobe XD":       ["adobe xd", "xd"],
    "UX/UI":          ["ux", "ui", "user experience", "user interface", "ux/ui", "ui/ux"],
    "Premiere Pro":   ["premiere", "adobe premiere"],
    "After Effects":  ["after effects"],
    "Canva":          ["canva"],

    # ── Testing & QA ──
    "Testing":        ["testing", "qa", "quality assurance", "test automation",
                       "selenium", "cypress", "pytest", "junit", "appium"],

    # ── Tools & Practices ──
    "Git":            ["git", "github", "gitlab", "bitbucket", "version control"],
    "Agile/Scrum":    ["agile", "scrum", "kanban", "jira", "sprint"],
    "OOP":            ["oop", "object oriented", "lập trình hướng đối tượng"],
    "Design Patterns":["design pattern", "solid", "clean code", "clean architecture"],
    "Excel":          ["excel", "google sheets", "spreadsheet"],

    # ── Soft Skills ──
    "English":              ["english", "tiếng anh", "ielts", "toeic"],
    "Giao tiếp":            ["giao tiếp", "communication", "trình bày", "thuyết trình"],
    "Làm việc nhóm":        ["làm việc nhóm", "teamwork", "phối hợp nhóm"],
    "Tư duy phản biện":     ["tư duy phản biện", "critical thinking", "problem solving",
                              "giải quyết vấn đề"],
    "Quản lý thời gian":    ["quản lý thời gian", "time management"],
    "Tự học":               ["tự học", "self-learning", "ham học hỏi", "cầu tiến"],
    "Quản lý dự án":        ["quản lý dự án", "project management", "pm"],
}

# Tạo reverse map: pattern → canonical_name (sắp xếp theo độ dài giảm dần)
_PATTERN_MAP: list[tuple[str, str]] = []
for _skill, _aliases in SKILL_DICT.items():
    for _alias in _aliases:
        _PATTERN_MAP.append((_alias.lower(), _skill))
_PATTERN_MAP.sort(key=lambda x: len(x[0]), reverse=True)


def _strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    s   = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return s.replace("\u0111", "d").replace("\u0110", "D")


def extract_skills_from_text(text: str) -> list[str]:
    """Extract skill names từ text bằng keyword matching."""
    if not text:
        return []
    normalized = " " + _strip_accents(text).lower() + " "
    found = set()
    for pattern, skill in _PATTERN_MAP:
        if f" {pattern} " in normalized or f",{pattern}," in normalized:
            found.add(skill)
    return list(found)


def _parse_salary_to_million(salary_raw: str) -> float:
    """Parse salary_raw → float triệu. Trả 0 nếu không parse được."""
    if not salary_raw:
        return 0.0
    s = salary_raw.lower()
    m = re.search(r"([\d,.]+)\s*[-–]\s*([\d,.]+)", s)
    if m:
        lo = float(m.group(1).replace(",", "."))
        hi = float(m.group(2).replace(",", "."))
        avg = (lo + hi) / 2
        return avg if avg < 500 else avg / 1_000_000  # nếu quá lớn → đổi đơn vị
    return 0.0


# ── SkillExtractor ────────────────────────────────────────────────────────────

class SkillExtractor:
    """
    Xây dựng skill taxonomy từ Qdrant collection.

    Dùng:
        extractor = SkillExtractor(qdrant_client, "topcv_jobs_v3")
        taxonomy  = extractor.get_taxonomy()
        # taxonomy[skill] = {count, experience_levels, avg_salary_min, jobs[]}
    """

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client          = client
        self.collection_name = collection_name

    def get_taxonomy(self) -> dict:
        """Load từ cache hoặc build nếu chưa có / stale."""
        if os.path.exists(_TAXONOMY_PATH):
            try:
                with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                total = self.client.count(self.collection_name).count
                if cached.get("_meta", {}).get("vector_count") == total:
                    print(f"[SkillExtractor] Loaded from cache | "
                          f"{len(cached)-1} skills | {total:,} vectors")
                    return {k: v for k, v in cached.items() if k != "_meta"}
                else:
                    print("[SkillExtractor] Cache stale — rebuilding...")
            except Exception as e:
                print(f"[SkillExtractor] Cache invalid ({e}) — rebuilding...")

        return self.build()

    def build(self) -> dict:
        """Scroll toàn bộ jobs, extract skills, tính taxonomy."""
        print("[SkillExtractor] Building taxonomy...", end=" ", flush=True)
        t0 = time.time()

        taxonomy: dict[str, dict] = {}
        n_jobs = 0
        offset = None

        while True:
            try:
                batch, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="section", match=MatchValue(value="requirements")),
                    ]),
                    limit=500,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                print(f"\n[SkillExtractor] Scroll error: {e}")
                break

            if not batch:
                break

            for point in batch:
                pl      = point.payload
                req_text = pl.get("section_text", "") or ""
                # Thêm title để catch skill từ tên job
                title   = pl.get("title", "") or ""
                text    = f"{title} {req_text}"

                skills = extract_skills_from_text(text)
                if not skills:
                    continue

                salary_min = pl.get("salary_min", 0.0) or 0.0
                level      = (pl.get("level", "") or "").lower()
                job_id     = pl.get("job_id", "")
                job_url    = pl.get("url", "")
                job_title  = pl.get("title", "")

                for skill in skills:
                    if skill not in taxonomy:
                        taxonomy[skill] = {
                            "count":            0,
                            "experience_levels": {},
                            "salary_sum":        0.0,
                            "salary_count":      0,
                            "avg_salary_min":    0.0,
                            "sample_jobs":       [],
                        }
                    s = taxonomy[skill]
                    s["count"] += 1

                    # Track experience levels
                    lvl_key = "fresher" if "fresher" in level else (
                        "junior" if "junior" in level else (
                            "senior" if any(k in level for k in ["senior", "lead", "manager"]) else "other"
                        )
                    )
                    s["experience_levels"][lvl_key] = s["experience_levels"].get(lvl_key, 0) + 1

                    # Track salary
                    if salary_min > 0:
                        s["salary_sum"]   += salary_min
                        s["salary_count"] += 1

                    # Giữ tối đa 5 sample jobs
                    if len(s["sample_jobs"]) < 5 and job_url:
                        s["sample_jobs"].append({
                            "title": job_title,
                            "url":   job_url,
                            "job_id": job_id,
                        })

                n_jobs += 1

            if offset is None:
                break

        # Tính avg_salary_min
        for skill, data in taxonomy.items():
            if data["salary_count"] > 0:
                data["avg_salary_min"] = round(data["salary_sum"] / data["salary_count"], 1)
            del data["salary_sum"]
            del data["salary_count"]

        elapsed = time.time() - t0
        print(f"{len(taxonomy)} skills | {n_jobs:,} jobs | {elapsed:.1f}s")

        # Save cache
        try:
            total = self.client.count(self.collection_name).count
            os.makedirs(os.path.dirname(_TAXONOMY_PATH), exist_ok=True)
            with open(_TAXONOMY_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {"_meta": {"vector_count": total}, **taxonomy},
                    f, ensure_ascii=False, indent=2,
                )
            print(f"[SkillExtractor] Cache saved → {_TAXONOMY_PATH}")
        except Exception as e:
            print(f"[SkillExtractor] Cache save failed: {e}")

        return taxonomy
