"""
hf_client.py — Groq API Client v6
====================================
Production-ready Groq HTTP client với retry thông minh.

Cải tiến:
  [v4-FIX-1] Default max_tokens: 1024 → 800
  [v4-FIX-2] 429 đọc Retry-After header chính xác
  [v4-FIX-3] Retries: 4 → 3
  [v4-FIX-4] Tách 429 logic riêng khỏi 5xx

Khuyến nghị max_tokens theo use case:
  job_search:      1000 (liệt kê 3 jobs)
  career_advice:   1400 (tư vấn có giải thích)
  career_guidance: 1600 (roadmap đầy đủ)
  chitchat:         300 (1-2 câu)
  query rewrite:    150 (1 câu)
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = os.getenv("GROQ_API_KEYS", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TIMEOUT  = int(os.getenv("GROQ_TIMEOUT", 60))
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

_SERVER_ERROR_CODES = {500, 502, 503, 529}


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _GroqResponse:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class GroqClient:
    """
    Groq HTTP client — production-ready với retry thông minh.

    Dùng:
        llm  = GroqClient()
        resp = llm.chat(messages=[...], max_tokens=1000)
        text = resp.choices[0].message.content
    """

    def __init__(self, api_keys: str | None = None, model: str = GROQ_MODEL):
        if not api_keys:
            api_keys = GROQ_API_KEYS or GROQ_API_KEY

        if not api_keys or api_keys in ("your_groq_api_key_here", ""):
            raise ValueError(
                "Thiếu GROQ_API_KEYS hoặc GROQ_API_KEY trong .env\n"
                "Lấy key miễn phí: https://console.groq.com/keys"
            )

        self.api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
        self._key_idx = 0
        self.model    = model
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_keys[self._key_idx]}",
            "Content-Type":  "application/json",
        })
        print(f"[Groq] model={self.model} | {len(self.api_keys)} keys loaded | timeout={GROQ_TIMEOUT}s")

    def _rotate_key(self):
        """Xoay vòng sang API key tiếp theo."""
        self._key_idx = (self._key_idx + 1) % len(self.api_keys)
        self._session.headers.update({"Authorization": f"Bearer {self.api_keys[self._key_idx]}"})
        print(f"[Groq] Key rotated → {self._key_idx + 1}/{len(self.api_keys)}")

    def chat(
        self,
        messages:    list[dict],
        max_tokens:  int   = 800,
        temperature: float = 0.5,
        retries:     int   = 3,
    ) -> _GroqResponse:
        """Gọi Groq API với retry thông minh."""
        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

        for attempt in range(retries + 1):
            try:
                resp = self._session.post(GROQ_ENDPOINT, json=payload, timeout=GROQ_TIMEOUT)

                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    return _GroqResponse(content)

                # Rate limit (429) — rotate key nếu có, sau đó backoff
                if resp.status_code == 429:
                    if len(self.api_keys) > 1:
                        self._rotate_key()
                    if attempt >= retries:
                        raise RuntimeError(f"Groq rate limit sau {retries} retries")
                    wait = float(resp.headers.get("Retry-After") or 10 * (2 ** attempt))
                    print(f"[Groq] 429 Rate limit — chờ {wait:.0f}s (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                    continue

                # Server errors (5xx)
                if resp.status_code in _SERVER_ERROR_CODES:
                    if attempt >= retries:
                        raise RuntimeError(f"Groq server error {resp.status_code}")
                    wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                    print(f"[Groq] {resp.status_code} — retry {attempt+1}/{retries} sau {wait}s")
                    time.sleep(wait)
                    continue

                # Lỗi không retry được
                raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:200]}")

            except requests.Timeout:
                if attempt >= retries:
                    raise RuntimeError(f"Groq timeout sau {retries} retries")
                wait = 5 * (2 ** attempt)
                print(f"[Groq] Timeout — retry {attempt+1}/{retries} sau {wait}s")
                time.sleep(wait)

            except requests.ConnectionError as e:
                if attempt >= retries:
                    raise RuntimeError(f"Groq connection error: {e}")
                time.sleep(5)

        raise RuntimeError("Groq chat: exceeded retries")

    def close(self):
        """Đóng session."""
        try:
            self._session.close()
        except Exception:
            pass


# Backward-compat alias
HFClient = GroqClient
