"""
career_guidance/market_analyzer.py — Market Intelligence Layer v6
===================================================================
Tổng hợp và truy vấn market data từ skill taxonomy.

Cung cấp:
  - top_skills_overall()      → top N skills toàn thị trường
  - top_skills_for_field()    → top N skills theo lĩnh vực
  - skill_demand_for_role()   → skills phổ biến cho 1 vị trí
  - salary_insight()          → salary range theo skill set
  - gap_analysis()            → so sánh user skills vs thị trường
  - market_summary()          → text tóm tắt cho LLM prompt
"""

import re
import unicodedata

from .skill_extractor import SkillExtractor, SKILL_DICT, extract_skills_from_text


# ── Field → Skill Mapping ─────────────────────────────────────────────────────

FIELD_SKILL_GROUPS: dict[str, list[str]] = {
    "frontend":    ["React", "Vue.js", "Angular", "Next.js", "HTML/CSS", "JavaScript",
                    "TypeScript", "Redux", "Webpack", "UX/UI", "Figma"],
    "backend":     ["Python", "Java", "Node.js", "PHP", "Go", "Django", "Spring Boot",
                    "FastAPI", "Flask", "REST API", "SQL", "PostgreSQL", "MongoDB",
                    "Redis", "Docker", "Microservices"],
    "fullstack":   ["React", "Node.js", "TypeScript", "PostgreSQL", "MongoDB",
                    "Docker", "REST API", "Git", "HTML/CSS"],
    "mobile":      ["Flutter", "React Native", "Android", "iOS", "Dart", "Kotlin", "Swift"],
    "data":        ["Python", "SQL", "Pandas", "NumPy", "Power BI", "Tableau",
                    "ETL", "Data Analysis", "Excel", "Spark"],
    "ai_ml":       ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
                    "NLP", "Computer Vision", "scikit-learn", "Pandas", "NumPy", "LLM", "MLOps"],
    "devops":      ["Docker", "Kubernetes", "AWS", "GCP", "Azure", "CI/CD", "Linux",
                    "Terraform", "Git", "Python", "Kafka"],
    "design":      ["Figma", "Photoshop", "Illustrator", "Adobe XD", "UX/UI",
                    "Canva", "Premiere Pro", "After Effects"],
    "it_general":  ["Python", "SQL", "Git", "OOP", "Linux", "Docker", "REST API",
                    "Agile/Scrum", "JavaScript", "HTML/CSS"],
    "marketing":   ["Excel", "Canva", "English", "Giao tiếp", "Tư duy phản biện"],
    "qa_testing":  ["Testing", "Python", "Java", "SQL", "Git", "Agile/Scrum"],
}

# Map tên vị trí/ngành → field key
ROLE_FIELD_MAP: list[tuple[str, str]] = [
    ("frontend",         "frontend"),
    ("front-end",        "frontend"),
    ("react",            "frontend"),
    ("vue",              "frontend"),
    ("angular",          "frontend"),
    ("backend",          "backend"),
    ("back-end",         "backend"),
    ("java developer",   "backend"),
    ("python developer", "backend"),
    ("php developer",    "backend"),
    ("golang",           "backend"),
    ("fullstack",        "fullstack"),
    ("full-stack",       "fullstack"),
    ("full stack",       "fullstack"),
    ("mobile",           "mobile"),
    ("android",          "mobile"),
    ("ios",              "mobile"),
    ("flutter",          "mobile"),
    ("data analyst",     "data"),
    ("data engineer",    "data"),
    ("business analyst", "data"),
    ("data scientist",   "ai_ml"),
    ("machine learning", "ai_ml"),
    ("ai engineer",      "ai_ml"),
    ("deep learning",    "ai_ml"),
    ("llm",              "ai_ml"),
    ("mlops",            "devops"),
    ("devops",           "devops"),
    ("sre",              "devops"),
    ("cloud",            "devops"),
    ("qa",               "qa_testing"),
    ("tester",           "qa_testing"),
    ("kiểm thử",         "qa_testing"),
    ("designer",         "design"),
    ("thiết kế",         "design"),
    ("graphic",          "design"),
    ("ui/ux",            "design"),
    ("ux/ui",            "design"),
    ("lập trình",        "it_general"),
    ("software",         "it_general"),
    ("developer",        "it_general"),
    ("intern",           "it_general"),
    ("fresher",          "it_general"),
    ("marketing",        "marketing"),
    ("content",          "marketing"),
    ("seo",              "marketing"),
]


def _strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    s   = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return s.replace("\u0111", "d").replace("\u0110", "D")


def detect_field(text: str) -> str | None:
    """Phát hiện lĩnh vực từ text (tên vị trí, target role, ngành học)."""
    if not text:
        return None
    t = _strip_accents(text).lower()
    for keyword, field in ROLE_FIELD_MAP:
        kw_normalized = _strip_accents(keyword).lower()
        if kw_normalized in t:
            return field
    return None


# ── MarketAnalyzer ────────────────────────────────────────────────────────────

class MarketAnalyzer:
    """
    Phân tích thị trường dựa trên skill taxonomy.

    Dùng:
        analyzer = MarketAnalyzer(skill_extractor)
        analyzer.load()                             # build/load taxonomy
        top = analyzer.top_skills_for_field("backend", n=10)
        gap = analyzer.gap_analysis(["Python", "SQL"], "backend")
    """

    def __init__(self, skill_extractor: SkillExtractor):
        self.extractor = skill_extractor
        self._taxonomy: dict = {}

    def load(self):
        """Load hoặc build taxonomy."""
        self._taxonomy = self.extractor.get_taxonomy()
        print(f"[MarketAnalyzer] Ready | {len(self._taxonomy)} skills")
        return self

    def top_skills_overall(self, n: int = 20) -> list[dict]:
        """Top N skills toàn thị trường, sort theo count."""
        if not self._taxonomy:
            return []
        return sorted(
            [
                {"skill": k, **{k2: v2 for k2, v2 in v.items() if k2 != "sample_jobs"}}
                for k, v in self._taxonomy.items()
            ],
            key=lambda x: x["count"],
            reverse=True,
        )[:n]

    def top_skills_for_field(self, field: str, n: int = 10) -> list[dict]:
        """Top N skills cho một lĩnh vực cụ thể."""
        field_skills = FIELD_SKILL_GROUPS.get(field, [])
        if not field_skills:
            return self.top_skills_overall(n)

        results = []
        for skill in field_skills:
            if skill in self._taxonomy:
                data = self._taxonomy[skill]
                results.append({
                    "skill":          skill,
                    "count":          data["count"],
                    "avg_salary_min": data["avg_salary_min"],
                    "experience_levels": data.get("experience_levels", {}),
                })

        return sorted(results, key=lambda x: x["count"], reverse=True)[:n]

    def gap_analysis(
        self,
        user_skills: list[str],
        field: str | None = None,
        target_role: str | None = None,
    ) -> dict:
        """
        So sánh kỹ năng user với thị trường → skill gap.

        Returns:
            {
                field: str,
                field_skills: list[str],    # skills thị trường cần
                user_skills: list[str],     # skills user có
                matched_skills: list[str],  # intersection
                missing_skills: list[dict], # thiếu, sort by count desc
                coverage_pct: int,          # % coverage
            }
        """
        if not field and target_role:
            field = detect_field(target_role)
        if not field:
            field = "it_general"

        field_skills = FIELD_SKILL_GROUPS.get(field, list(SKILL_DICT.keys())[:20])

        # Normalize user skills (so sánh case-insensitive)
        user_set = {_strip_accents(s).lower() for s in (user_skills or [])}

        matched = []
        missing = []
        for skill in field_skills:
            skill_norm = _strip_accents(skill).lower()
            if skill_norm in user_set or any(skill_norm in _strip_accents(u).lower() for u in user_set):
                matched.append(skill)
            else:
                skill_data = self._taxonomy.get(skill, {})
                missing.append({
                    "skill":          skill,
                    "count":          skill_data.get("count", 0),
                    "avg_salary_min": skill_data.get("avg_salary_min", 0),
                })

        missing.sort(key=lambda x: x["count"], reverse=True)
        coverage = int(len(matched) / max(len(field_skills), 1) * 100)

        return {
            "field":          field,
            "field_skills":   field_skills,
            "user_skills":    user_skills or [],
            "matched_skills": matched,
            "missing_skills": missing,
            "coverage_pct":   coverage,
        }

    def salary_insight(self, skills: list[str]) -> dict:
        """Salary insight cho một tập skills."""
        salaries = []
        for skill in skills:
            data = self._taxonomy.get(skill)
            if data and data.get("avg_salary_min", 0) > 0:
                salaries.append(data["avg_salary_min"])

        if not salaries:
            return {"min": 0, "max": 0, "avg": 0}

        return {
            "min": round(min(salaries), 1),
            "max": round(max(salaries), 1),
            "avg": round(sum(salaries) / len(salaries), 1),
        }

    def market_summary(self, field: str, n: int = 8) -> str:
        """Tóm tắt thị trường dạng text cho LLM prompt."""
        top = self.top_skills_for_field(field, n=n)
        if not top:
            return f"Không có dữ liệu thị trường cho lĩnh vực '{field}'."

        lines = [f"📊 Market Data — {field.upper()}:"]
        for i, s in enumerate(top, 1):
            sal = f" | lương TB: {s['avg_salary_min']} triệu" if s.get("avg_salary_min") else ""
            lines.append(f"  {i}. {s['skill']}: {s['count']} job{sal}")

        return "\n".join(lines)

    def quick_market_insight(self, field: str) -> str:
        """Phiên bản ngắn gọn của market_summary — dùng trong career_advice."""
        top = self.top_skills_for_field(field, n=5)
        if not top:
            return ""
        skills_str = ", ".join(f"{s['skill']}({s['count']} jobs)" for s in top)
        return f"[MARKET INSIGHT — {field}] Top skills: {skills_str}"
