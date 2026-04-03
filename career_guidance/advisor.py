"""
career_guidance/advisor.py — Career Advisor v6
================================================
Kết hợp market data + user profile + RAG jobs → tư vấn định hướng cá nhân hóa.

Flow:
  1. Nhận user_profile (major, current_skills, target_role, experience_level)
  2. MarketAnalyzer.gap_analysis() → skill gap có cơ sở thực tế
  3. RAG.hybrid_search() → job minh họa phù hợp với target_role
  4. Build LLM prompt → LLM generate roadmap cá nhân hóa
"""

from .market_analyzer import MarketAnalyzer, detect_field

SYSTEM_GUIDANCE = """\
Bạn là chuyên gia tư vấn định hướng nghề nghiệp của TopCV, dựa trên phân tích \
dữ liệu thị trường lao động thực tế.

NHIỆM VỤ: Tư vấn định hướng nghề nghiệp CÁ NHÂN HÓA dựa trên:
  (1) Thông tin hồ sơ sinh viên
  (2) Dữ liệu market thực tế từ hàng nghìn bài tuyển dụng
  (3) Danh sách việc làm minh họa phù hợp

QUY TẮC:
- KHÔNG bịa đặt thông tin ngoài [MARKET DATA] và [JOB DATA]
- Dùng số liệu cụ thể từ market data (X job yêu cầu skill Y)
- Trả lời bằng tiếng Việt, tối đa 500 từ
- Tuyệt đối KHÔNG dùng Heading Markdown (#, ##, ###)

CẤU TRÚC TRẢ LỜI BẮT BUỘC:

【Đánh giá Profile】
Nhận xét khách quan điểm mạnh/yếu. Tỉ lệ coverage: X% (% skills đã có so với thị trường).

【Skill Gap — Ưu tiên học ngay】
Liệt kê 3-5 skills quan trọng nhất còn thiếu. Mỗi skill: lý do cần + ước lượng thời gian học.

【Lộ trình 6 tháng】
Chia theo tháng (Tháng 1-2, 3-4, 5-6). Cụ thể, thực hiện được.

【Cơ hội nghề nghiệp】
Mức lương thực tế từ market data. Job phù hợp nhất để apply sau 6 tháng.

【Việc làm tham khảo】
Tối đa 3 job từ [JOB DATA]:
- **[Tên vị trí]** — [Lương]: [Chi tiết công việc](link)
"""


def _parse_skills_from_input(text: str) -> list[str]:
    """Parse kỹ năng từ input tự do của user."""
    from .skill_extractor import extract_skills_from_text
    # Thử extract qua skill dict trước
    extracted = extract_skills_from_text(text)
    if extracted:
        return extracted
    # Fallback: split theo dấu phẩy/chấm phẩy
    raw = [s.strip() for s in text.replace(";", ",").replace("/", ",").split(",")]
    return [s for s in raw if 2 <= len(s) <= 50]


def _detect_experience_level(text: str) -> str:
    """Detect experience level từ text tự do."""
    t = text.lower()
    if any(k in t for k in ["fresher", "mới ra trường", "chưa có kinh nghiệm", "sinh viên"]):
        return "fresher"
    if any(k in t for k in ["junior", "1 năm", "2 năm", "mới đi làm"]):
        return "junior"
    if any(k in t for k in ["senior", "3 năm", "4 năm", "5 năm", "nhiều năm"]):
        return "senior"
    return "other"


class CareerAdvisor:
    """
    Tư vấn định hướng nghề nghiệp cá nhân hóa.

    Dùng:
        advisor = CareerAdvisor(market_analyzer, rag, llm)
        result  = advisor.generate_guidance(profile, query)
        # result = {system, context, valid_links, gap, field}
    """

    def __init__(self, market_analyzer: MarketAnalyzer, rag, llm):
        self.market = market_analyzer
        self.rag    = rag
        self.llm    = llm

    def generate_guidance(
        self,
        user_profile: dict,
        query: str,
        embedding_model=None,
    ) -> dict:
        """
        Tạo tư vấn định hướng dựa trên profile + market data + RAG.

        Args:
            user_profile: {
                major:            str  — ngành học / lĩnh vực
                current_skills:   list — kỹ năng hiện có
                target_role:      str  — vị trí muốn hướng tới
                experience_level: str  — fresher / junior / other
            }
        Returns:
            {system, context, valid_links, gap, field}
        """
        major          = user_profile.get("major", "")
        current_skills = user_profile.get("current_skills", [])
        target_role    = user_profile.get("target_role", "")
        exp_level      = user_profile.get("experience_level", "other")

        # Detect field từ target_role hoặc major
        field = detect_field(target_role) or detect_field(major) or "it_general"

        # Gap analysis
        gap = self.market.gap_analysis(
            user_skills=current_skills,
            field=field,
            target_role=target_role,
        )

        # Market summary
        market_text = self.market.market_summary(field, n=10)

        # RAG: tìm job minh họa phù hợp với target_role
        search_query = target_role or major or query
        valid_links  = set()
        jobs_context = "Không tìm được việc làm minh họa."

        if embedding_model and search_query:
            try:
                query_vec = embedding_model.get_query_embedding(search_query)
                jobs      = self.rag.hybrid_search(
                    search_query, query_vec,
                    filters={},
                    limit=3,
                )
                if jobs:
                    jobs_context, valid_links = self.rag.enhance_prompt(
                        search_query, query_vec, jobs=jobs
                    )
            except Exception as e:
                print(f"[Advisor] RAG error: {e}")

        # Build context string
        context = self._build_context(
            user_profile=user_profile,
            gap=gap,
            market_text=market_text,
            jobs_context=jobs_context,
            exp_level=exp_level,
        )

        return {
            "system":      SYSTEM_GUIDANCE,
            "context":     context,
            "valid_links": valid_links,
            "gap":         gap,
            "field":       field,
        }

    def _build_context(
        self,
        user_profile: dict,
        gap: dict,
        market_text: str,
        jobs_context: str,
        exp_level: str,
    ) -> str:
        major     = user_profile.get("major", "N/A")
        skills    = user_profile.get("current_skills", [])
        target    = user_profile.get("target_role", "N/A")
        field     = gap.get("field", "it_general")

        skills_str  = ", ".join(skills) if skills else "Chưa có"
        matched_str = ", ".join(gap.get("matched_skills", [])) or "Không có"
        missing     = gap.get("missing_skills", [])[:5]
        missing_str = "\n".join(
            f"  - {m['skill']}: xuất hiện trong {m['count']} job"
            + (f" | lương TB {m['avg_salary_min']} triệu" if m.get("avg_salary_min") else "")
            for m in missing
        ) or "  (Không có dữ liệu)"

        return f"""\
[HỒ SƠ NGƯỜI DÙNG]
Ngành học / Chuyên môn : {major}
Kỹ năng hiện có        : {skills_str}
Vị trí mục tiêu        : {target}
Cấp độ                 : {exp_level}
Lĩnh vực phát hiện     : {field}

[SKILL GAP ANALYSIS]
Coverage: {gap.get('coverage_pct', 0)}%
Kỹ năng đã có (match market): {matched_str}
Kỹ năng còn thiếu (quan trọng nhất):
{missing_str}

[MARKET DATA]
{market_text}

[JOB DATA — Việc làm minh họa]
{jobs_context}"""

    def quick_market_insight(self, field: str) -> str:
        """Quick insight cho career_advice route."""
        return self.market.quick_market_insight(field)
