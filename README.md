# 🤖 Career Bot v6.3

Chatbot tư vấn việc làm thông minh — tìm kiếm, lọc, và gợi ý công việc phù hợp dựa trên dữ liệu thực tế từ TopCV.

## Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc Pipeline](#kiến-trúc-pipeline)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Các module chính](#các-module-chính)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
- [Cấu hình .env](#cấu-hình-env)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [API Endpoints](#api-endpoints)
- [System Prompts](#system-prompts)
- [Đánh giá (Evaluation)](#đánh-giá-evaluation)
- [Changelog](#changelog)

---

## Tổng quan

Career Bot là hệ thống RAG (Retrieval-Augmented Generation) multi-stage, kết hợp:

| Thành phần | Công nghệ | Vai trò |
|-----------|-----------|---------|
| **Embedding** | Jina Embeddings v3 (1024 dims) | Encode query → vector |
| **Vector DB** | Qdrant Cloud | Lưu trữ & tìm kiếm semantic |
| **BM25** | rank-bm25 (Okapi) | Keyword search song song |
| **Reranker** | Qwen3-Reranker-0.6B | Cross-encoder rerank top candidates |
| **LLM** | Groq API (LLaMA 3.1 8B) | Sinh câu trả lời cuối cùng |
| **Intent Router** | Semantic Router (cosine similarity) | Phân loại intent <5ms |
| **History** | Qdrant + RAM Cache | Lưu hội thoại, hỗ trợ query rewrite |

---

## Kiến trúc Pipeline

```
User Query
    │
    ▼
┌──────────────────┐
│  Semantic Router  │  ← Phân loại intent (job_search / career_advice / chitchat)
│  (cosine sim)     │     Confidence < 0.35 → chitchat
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 chitchat   job_search / career_advice
    │         │
    │    ┌────┴────────────────────┐
    │    │  Smart Rewrite          │  ← Nếu query ngắn/thiếu ngữ cảnh
    │    │  (LLM query rewrite)    │     → viết lại thành standalone query
    │    └────┬────────────────────┘
    │         │
    │    ┌────┴────────────────────┐
    │    │  Query Parser            │  ← Trích xuất filter:
    │    │  _parse_query()          │     • location_norm (HN, HCM, ĐN...)
    │    │                          │     • salary_min (≥ X triệu)
    │    │                          │     • experience_norm (fresher/junior/senior)
    │    │                          │     • link_only (chỉ trả link)
    │    └────┬────────────────────┘
    │         │
    │    ┌────┴────────────────────┐
    │    │  Context Merge           │  ← Merge filter từ session trước
    │    │  _merge_ctx()            │     (hỗ trợ hội thoại đa lượt)
    │    └────┬────────────────────┘
    │         │
    │    ┌────┴────────────────────────────────────┐
    │    │  Hybrid Search (RAG)                     │
    │    │  ┌─────────────┐  ┌─────────────┐       │
    │    │  │ Vector Search│  │ BM25 Search │       │
    │    │  │ (Qdrant)     │  │ (rank-bm25) │       │
    │    │  └──────┬──────┘  └──────┬──────┘       │
    │    │         └───────┬────────┘               │
    │    │                 ▼                        │
    │    │          RRF Fusion                      │
    │    │         (w=0.6/0.4)                      │
    │    │                 │                        │
    │    │                 ▼                        │
    │    │     Load Full Jobs (Qdrant scroll)       │
    │    │                 │                        │
    │    │                 ▼                        │
    │    │     Qwen3 Reranker (cross-encoder)       │
    │    │         P("yes") scoring                 │
    │    └────┬────────────────────────────────────┘
    │         │
    │    ┌────┴────────────────────┐
    │    │  Build RAG Messages      │  ← System prompt + filter hints
    │    │  + enhance_prompt()      │     + [DỮ LIỆU VIỆC LÀM]
    │    │  + [BỘ LỌC USER YÊU CẦU]│     + [LINK HỢP LỆ]
    │    └────┬────────────────────┘
    │         │
    └────┬────┘
         │
    ┌────┴────────────────────┐
    │  Groq LLM                │  ← Sinh câu trả lời
    │  (LLaMA 3.1 8B Instant)  │     max_tokens/temperature theo route
    └────┬────────────────────┘
         │
    ┌────┴────────────────────┐
    │  Post-processing         │
    │  • _filter_links()       │  ← Xoá link bịa (không có trong data)
    │  • _ensure_format()      │  ← Xoá pattern bẩn LLM tạo ra
    └────┬────────────────────┘
         │
    ┌────┴────────────────────┐
    │  Save History (async)    │  ← RAM cache + Qdrant (background thread)
    └────┬────────────────────┘
         │
         ▼
    JSON Response
    { content, route, confidence }
```

---

## Cấu trúc thư mục

```
career_bot_v6/
├── flask_serve.py              # API server chính (Flask)
├── demo_app.py                 # Streamlit demo UI
├── prompts.py                  # System prompts (v6.3)
├── hf_client.py                # Groq LLM client (HTTP + retry)
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Deadline cleaner (Docker)
├── .env                        # API keys & config (không commit)
│
├── embedding_model/
│   ├── __init__.py
│   └── core.py                 # Jina v3 embedding wrapper
│
├── rag/
│   ├── __init__.py
│   └── core.py                 # Hybrid search + BM25 + Qwen3 reranker
│
├── semantic_router/
│   ├── __init__.py
│   └── router.py               # Intent classification (cosine sim)
│
├── reflection/
│   ├── __init__.py
│   └── core.py                 # Chat history + query rewriting
│
├── pipeline/
│   └── deadline_cleaner.py     # Tự động xoá job hết hạn
│
├── scripts/
│   └── crawl.py                # TopCV crawler (Selenium)
│
├── data/
│   ├── TopCV_Jobs_Data.jsonl   # Raw crawled data (~3.3MB)
│   ├── intents.json            # Intent samples (job_search, career_advice, chitchat)
│   ├── vn_stopwords.txt        # Vietnamese stopwords cho BM25
│   ├── bm25_cache.pkl          # BM25 index cache (auto-generated)
│   └── cleaner_logs/           # Audit logs từ deadline cleaner
│
├── test/
│   ├── test_retrieval_50q.csv  # 50 test queries với expected job IDs
│   └── human_eval_queries.csv  # Human evaluation queries
│
├── eval_jina_reranker.py       # Recall@K evaluation (Jina reranker)
├── eval_qwen3_reranker.py      # Recall@K evaluation (Qwen3 reranker)
└── generate_human_eval.py      # Tạo dữ liệu human eval từ API
```

---

## Các module chính

### `flask_serve.py` — API Server

Entrypoint chính. Xử lý toàn bộ luồng từ nhận query → trả kết quả.

| Component | Chức năng |
|-----------|----------|
| `get_pipeline()` | Lazy init tất cả components (thread-safe, singleton) |
| `_HistoryCache` | RAM cache conversation history (TTL 5 phút, max 20 msgs) |
| `_CTX_STORE` | Session context: keyword, filters, seen_ids (TTL 2 giờ) |
| `_parse_query()` | Trích xuất location, salary, experience, link_only từ query |
| `_merge_ctx()` | Merge filter mới với context cũ (hỗ trợ "lương cao hơn") |
| `_smart_rewrite()` | Viết lại query nếu quá ngắn/thiếu nghĩa |
| `_build_rag_messages()` | Build messages cho LLM: system + filter hints + RAG context |
| `_ensure_format()` | Post-processing: xoá headings, pattern bẩn |
| `_filter_links()` | Xoá link bịa không có trong data |

### `rag/core.py` — Hybrid Search Engine

| Method | Chức năng |
|--------|----------|
| `hybrid_search()` | Vector + BM25 → RRF fusion → rerank |
| `vector_search()` | Qdrant semantic search (cosine, threshold 0.25) |
| `bm25_search()` | BM25Okapi search (Vietnamese tokenized, stopword removed) |
| `_rerank()` | Qwen3-Reranker-0.6B: P("yes") scoring |
| `enhance_prompt()` | Format jobs → LLM context (clean N/A, truncate) |
| `_build_filter()` | Qdrant filter: location, salary, deadline, experience |

### `semantic_router/router.py` — Intent Classification

- 3 routes: `job_search`, `career_advice`, `chitchat`
- Cosine similarity giữa query embedding và pre-built route embeddings
- Threshold 0.35 → dưới ngưỡng = chitchat
- Inference <5ms sau warmup

### `reflection/core.py` — History & Query Rewriting

- Lưu hội thoại vào Qdrant collection riêng (`career_chat_history`)
- Rolling window 10 turns (20 messages)
- Query rewriting: viết lại câu hỏi ngắn thành standalone query đầy đủ nghĩa

### `hf_client.py` — Groq LLM Client

- HTTP client thuần (không dùng SDK)
- Hỗ trợ multiple API keys + auto-rotate khi 429
- Retry logic: 429 (rate limit), 5xx (server error), timeout
- 3 retries, exponential backoff

### `prompts.py` — System Prompts (v6.3)

4 system prompts cho 4 routes:

| Prompt | Route | Đặc điểm chính |
|--------|-------|----------------|
| `SYSTEM_JOB_RAG` | job_search | Filter enforcement (lương/KN/địa điểm), format ①②③, cấm bịa/lặp job |
| `SYSTEM_LINK_ONLY` | link_only | Chỉ trả link, không giải thích |
| `SYSTEM_ADVICE_RAG` | career_advice | Tư vấn + kỹ năng đi kèm, chặn liệt kê job khi không cần |
| `SYSTEM_CHAT` | chitchat | Thân thiện, tối đa 4 câu |

---

## Hướng dẫn cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- 8GB+ RAM (cho Jina v3 + Qwen3 reranker)
- Qdrant Cloud account (miễn phí)
- Groq API key (miễn phí)

### Bước 1: Clone & cài dependencies

```bash
git clone https://github.com/LDH-LeHieu18012005/Career-Bot.git
cd Career-Bot

# Tạo virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### Bước 2: Download embedding model (offline)

Jina v3 chạy local, cần download trước:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
```

### Bước 3: Download Qwen3 Reranker (offline)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
```

### Bước 4: Cấu hình `.env`

Copy `.env.example` (nếu có) hoặc tạo file `.env` với nội dung bên dưới.

### Bước 5: Ingest dữ liệu vào Qdrant

Chạy notebook `qrantdtbs (2).ipynb` trên Kaggle (có GPU) để:
1. Đọc `TopCV_Jobs_Data.jsonl`
2. Chunk theo sections (overview, description, requirements, benefits)
3. Embed với Jina v3
4. Upload vectors lên Qdrant Cloud

---

## Cấu hình .env

```env
# ── Groq LLM ─────────────────────────────────────────────────
# Nhiều key cách nhau bởi dấu phẩy → auto-rotate khi 429
GROQ_API_KEYS=gsk_key1,gsk_key2
GROQ_MODEL=llama-3.1-8b-instant

# ── Qdrant Cloud ─────────────────────────────────────────────
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=topcv_jobs_v3

# ── Embedding (local) ───────────────────────────────────────
EMBED_MODEL=jinaai/jina-embeddings-v3
EMBED_DEVICE=           # trống = auto (cuda > mps > cpu)

# ── Reranker ─────────────────────────────────────────────────
QWEN3_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
QWEN3_RERANKER_DEVICE=  # trống = auto
HF_TOKEN=hf_your_token  # để download Qwen3 (cần accept license)

# ── RAG ──────────────────────────────────────────────────────
DEFAULT_SEARCH_LIMIT=5
SIMILARITY_THRESHOLD=0.25
RERANK_TOP_K=10

# ── Flask ────────────────────────────────────────────────────
FLASK_PORT=5001
```

---

## Chạy ứng dụng

### Chạy API Server

```bash
python flask_serve.py
```

Server khởi động tại `http://localhost:5001`. Quá trình warmup:
1. Xoá job hết hạn (deadline cleaner)
2. Load Jina v3 embedding model
3. Connect Qdrant + build BM25 index
4. Load Qwen3 reranker (background thread)
5. Build semantic router embeddings

### Chạy Demo UI (Streamlit)

```bash
# Đảm bảo Flask server đang chạy
pip install streamlit
streamlit run demo_app.py
```

### Chạy Deadline Cleaner (Docker)

```bash
docker-compose up -d deadline_cleaner
```

Tự động xoá job hết hạn mỗi ngày lúc 02:00 UTC.

### Crawl dữ liệu mới

```bash
python scripts/crawl.py --target 5000 --category it --headless
```

---

## API Endpoints

### `GET /api/v1/health`

Kiểm tra trạng thái server.

**Response:**
```json
{
  "status": "ok",
  "rag_count": 12500,
  "bm25_status": { "ready": true, "docs": 2500 }
}
```

### `POST /api/v1/chat`

Gửi câu hỏi và nhận phản hồi.

**Request:**
```json
{
  "query": "tìm việc python lương trên 15 triệu hà nội",
  "session_id": "uuid-session-id"
}
```

**Response:**
```json
{
  "content": "① **Backend Developer Python** — 15 - 25 triệu\n   ...",
  "route": "job_search",
  "confidence": 0.847
}
```

**Chi tiết các route:**

| Route | Trigger | LLM Config | Prompt |
|-------|---------|-----------|--------|
| `job_search` | "tìm việc", "tuyển dụng", "việc làm" | max_tokens=1200, temp=0.2 | `SYSTEM_JOB_RAG` |
| `career_advice` | "tư vấn", "lộ trình", "kỹ năng" | max_tokens=1600, temp=0.3 | `SYSTEM_ADVICE_RAG` |
| `chitchat` | "xin chào", "cảm ơn", fallback | max_tokens=400, temp=0.7 | `SYSTEM_CHAT` |
| `link_only` | "link", "ứng tuyển", "apply" (auto-detect) | max_tokens=600, temp=0.0 | `SYSTEM_LINK_ONLY` |

---

## System Prompts

### SYSTEM_JOB_RAG (v6.3)

Cải tiến so với v6:

| Tính năng | Mô tả |
|-----------|-------|
| **Filter enforcement** | Quy tắc lọc BẮT BUỘC cho lương, KN, địa điểm — LLM không được bỏ qua |
| **Xếp hạng ưu tiên** | Job phù hợp nhất hiển thị ĐẦU TIÊN (vị trí → lương → KN → địa điểm) |
| **Format cứng ①②③** | Delimiter rõ ràng, LLM khó phá format |
| **Câu nhận xét cụ thể** | Phải nêu rõ bao nhiêu job đáp ứng, tiêu chí nào chưa khớp |
| **Chống bịa job** | Cấm tạo job không có trong data |
| **Chống lặp job** | Cấm hiển thị cùng job trùng lặp trong 1 câu trả lời |
| **Filter hints** | `_build_rag_messages()` truyền `[BỘ LỌC USER YÊU CẦU]` cụ thể cho LLM |

### SYSTEM_ADVICE_RAG (v6.3)

| Tính năng | Mô tả |
|-----------|-------|
| **Chặn liệt kê job** | KHÔNG liệt kê job khi user hỏi lộ trình/CV/phỏng vấn/chứng chỉ |
| **Kỹ năng đi kèm** | Nêu kỹ năng cụ thể theo dữ liệu (Backend: Python/Java/SQL/Docker...) |
| **Trích dẫn số liệu** | Dùng mức lương, kỹ năng, cấp bậc từ data thực tế |

---

## Đánh giá (Evaluation)

### Retrieval Evaluation

```bash
# Đánh giá với Qwen3 reranker
python eval_qwen3_reranker.py

# Đánh giá với Jina reranker
python eval_jina_reranker.py

# Chỉ RRF (không rerank)
python eval_qwen3_reranker.py --no-rerank
```

Metrics: Recall@1, Recall@5, Recall@10, Hit@1/5/10 → phân tích theo difficulty (easy/medium/hard).

**Kết quả đánh giá (50 queries):**

| Phương pháp (Reranker) | Recall@1 | Recall@5 | Recall@10 |
|------------------------|----------|----------|-----------|
| **Jina v3** | 0.337 | 0.597 | 0.713 |
| **Qwen 3 (0.6B)** | 0.360 | 0.663 | 0.733 |


### Human Evaluation

```bash
# Bước 1: Chạy Flask server
python flask_serve.py

# Bước 2: Tạo dữ liệu eval
python generate_human_eval.py # Đánh giá với Jina reranker
python generate_human_eval.py --reranker qwen3 # Đánh giá với Qwen3 reranker
```

---

## Changelog

### v6.3 (2026-04-13)

- **Prompt**: Viết lại `SYSTEM_JOB_RAG` với filter enforcement mạnh, format ①②③, cấm bịa/lặp job
- **Prompt**: Viết lại `SYSTEM_ADVICE_RAG` — chặn liệt kê job khi chỉ hỏi lời khuyên, thêm kỹ năng đi kèm
- **Server**: `_build_rag_messages()` truyền `[BỘ LỌC USER YÊU CẦU]` cụ thể cho LLM
- **Server**: `_ensure_format()` xoá 5 pattern bẩn LLM phổ biến

### v6.2

- **Server**: RAM cache conversation history (TTL 5 phút) → bỏ 1 Qdrant roundtrip/request
- **Server**: Async save_turn vẫn ghi Qdrant ở background thread
- **RAG**: Qwen3-Reranker-0.6B thay Jina reranker (nhẹ hơn, offline)

### v6.1

- **RAG**: Chuẩn hóa N/A → `_clean_na()` helper
- **RAG**: `enhance_prompt()` bỏ dòng Công ty nếu N/A
- **RAG**: Reranker bỏ "N/A" trong pair gửi cross-encoder

### v6.0

- **Architecture**: Semantic Router (cosine sim, 3 routes)
- **Search**: Hybrid search (Vector + BM25 + RRF fusion)
- **Embedding**: Jina v3 local (1024 dims, offline)
- **History**: Qdrant-based conversation history + query rewriting
- **Cleaner**: Tự động xoá job hết hạn (deadline_ts filter)

---

## License

MIT
