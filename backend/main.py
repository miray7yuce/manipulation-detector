from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import json
import os
import asyncio
import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    try:
        from readability import Document
        HAS_READABILITY = True
    except ImportError:
        HAS_READABILITY = False

try:
    from langdetect import detect as langdetect_detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

from openai import OpenAI


def _get_ollama_client():
    if OLLAMA_BASE_URL:
        return OpenAI(base_url=f"{OLLAMA_BASE_URL.rstrip('/')}/v1", api_key="ollama")
    return None


def _get_groq_client():
    if GROQ_API_KEY:
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
        )
    return None


ollama_client = _get_ollama_client()
groq_client = _get_groq_client()

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_client = genai
            return _gemini_client
        except ImportError:
            raise RuntimeError(
                "google-generativeai package not found. "
                "Run: pip install google-generativeai"
            )
    return None


# ── Known source bias database ────────────────────────────────
# Scores: -100 (far left) to +100 (far right), 0 = center
# Credibility: "high" | "mixed" | "low"
SOURCE_BIAS_DB: dict[str, dict] = {
    "bbc.com":          {"score": -5,  "label": "Center-Left",  "credibility": "high"},
    "bbc.co.uk":        {"score": -5,  "label": "Center-Left",  "credibility": "high"},
    "reuters.com":      {"score":  0,  "label": "Center",       "credibility": "high"},
    "apnews.com":       {"score":  0,  "label": "Center",       "credibility": "high"},
    "theguardian.com":  {"score": -30, "label": "Left",         "credibility": "high"},
    "nytimes.com":      {"score": -20, "label": "Center-Left",  "credibility": "high"},
    "washingtonpost.com":{"score":-18, "label": "Center-Left",  "credibility": "high"},
    "foxnews.com":      {"score":  40, "label": "Right",        "credibility": "mixed"},
    "breitbart.com":    {"score":  75, "label": "Far-Right",    "credibility": "low"},
    "cnn.com":          {"score": -22, "label": "Center-Left",  "credibility": "high"},
    "msnbc.com":        {"score": -35, "label": "Left",         "credibility": "mixed"},
    "wsj.com":          {"score":  15, "label": "Center-Right", "credibility": "high"},
    "economist.com":    {"score":  5,  "label": "Center-Right", "credibility": "high"},
    "huffpost.com":     {"score": -40, "label": "Left",         "credibility": "mixed"},
    "dailymail.co.uk":  {"score":  35, "label": "Right",        "credibility": "mixed"},
    "democracynow.org": {"score": -60, "label": "Far-Left",     "credibility": "mixed"},
    "theintercept.com": {"score": -45, "label": "Left",         "credibility": "mixed"},
    "nationalreview.com":{"score": 38, "label": "Right",        "credibility": "high"},
    "politico.com":     {"score": -10, "label": "Center-Left",  "credibility": "high"},
    "axios.com":        {"score":  -5, "label": "Center",       "credibility": "high"},
    # Turkish sources
    "cumhuriyet.com.tr":{"score": -30, "label": "Sol",          "credibility": "mixed"},
    "sabah.com.tr":     {"score":  35, "label": "Sağ",          "credibility": "mixed"},
    "hurriyet.com.tr":  {"score":   5, "label": "Merkez",       "credibility": "mixed"},
    "sozcu.com.tr":     {"score": -20, "label": "Merkez-Sol",   "credibility": "mixed"},
    "haberturk.com":    {"score":  10, "label": "Merkez-Sağ",   "credibility": "mixed"},
    "milliyet.com.tr":  {"score":  10, "label": "Merkez-Sağ",   "credibility": "mixed"},
    "bianet.org":       {"score": -35, "label": "Sol",          "credibility": "high"},
    "t24.com.tr":       {"score": -15, "label": "Merkez-Sol",   "credibility": "mixed"},
    "gazeteduvar.com.tr":{"score":-40, "label": "Sol",          "credibility": "mixed"},
    "independentturkish.com":{"score":0,"label":"Merkez",       "credibility": "high"},
}


def get_source_info(url: str | None) -> dict | None:
    if not url:
        return None
    url_lower = url.lower().replace("https://", "").replace("http://", "").replace("www.", "")
    for domain, info in SOURCE_BIAS_DB.items():
        if url_lower.startswith(domain) or f"/{domain}" in url_lower or url_lower == domain:
            return {"domain": domain, **info}
    # Try partial match (e.g. subdomain)
    for domain, info in SOURCE_BIAS_DB.items():
        if domain in url_lower:
            return {"domain": domain, **info}
    return None


# ── Language detection & prompts ──────────────────────────────
def detect_language(text: str) -> str:
    """Returns ISO 639-1 code, defaults to 'en' on failure."""
    if not HAS_LANGDETECT or not text:
        return "en"
    try:
        return langdetect_detect(text[:2000])
    except Exception:
        return "en"


SYSTEM_PROMPT_EN = """You are an expert media literacy analyst. Analyze news articles for bias and manipulation techniques.
Respond ONLY with valid JSON in this exact format (no markdown, no extra text, no code fences):
{
  "bias": "Brief description of detected bias (left/right/center, political slant, etc.) or 'Neutral' if balanced",
  "manipulation_techniques": ["List", "of", "techniques", "used"],
  "highlights": ["Key phrases or words that indicate manipulation or bias"],
  "confidence": 85,
  "explanation": "2-3 sentence explanation of your analysis"
}

The "confidence" field is an integer from 0 to 100 representing how confident you are in this analysis.
Common manipulation techniques: emotional language, selective framing, omission, loaded words, appeal to authority, false equivalence, cherry-picking, sensationalism, fear-mongering, straw man, ad hominem, etc."""

SYSTEM_PROMPT_TR = """Sen bir medya okuryazarlığı uzmanısın. Haber metinlerini önyargı ve manipülasyon teknikleri açısından analiz et.
SADECE aşağıdaki formatta geçerli JSON döndür (markdown, açıklama veya kod bloğu olmadan):
{
  "bias": "Tespit edilen önyargının kısa açıklaması (sol/sağ/merkez, siyasi eğilim vb.) veya tarafsızsa 'Tarafsız'",
  "manipulation_techniques": ["Kullanılan", "teknikler", "listesi"],
  "highlights": ["Manipülasyon veya önyargıya işaret eden anahtar ifadeler"],
  "confidence": 85,
  "explanation": "Analizinizin 2-3 cümlelik açıklaması"
}

"confidence" alanı, analizinize olan güven düzeyinizi 0-100 arasında bir tam sayı olarak ifade eder.
Yaygın manipülasyon teknikleri: duygusal dil, seçici çerçeveleme, atlama/eksik bilgi, yüklü kelimeler, otoriteye başvuru, yanlış denklik, kiraz toplama, sansasyonalizm, korku yaratma vb."""


def get_system_prompt(lang: str) -> str:
    return SYSTEM_PROMPT_TR if lang == "tr" else SYSTEM_PROMPT_EN


# ── JSON parser ───────────────────────────────────────────────
def _parse_llm_response(content: str) -> dict:
    content = (content or "").strip()
    if not content:
        raise ValueError("Model returned empty response")
    if "```" in content:
        parts = content.split("```")
        for part in parts[1:]:
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].lstrip()
            if part.strip().startswith("{"):
                content = part
                break
    if not content.strip().startswith("{"):
        start = content.find("{")
        if start >= 0:
            depth, end = 1, start + 1
            while end < len(content) and depth > 0:
                if content[end] == "{":
                    depth += 1
                elif content[end] == "}":
                    depth -= 1
                end += 1
            if depth == 0:
                content = content[start:end]
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse model response as JSON: {e}")


# ── URL text extraction ───────────────────────────────────────
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5,tr;q=0.3",
}

MAX_ARTICLE_CHARS = 8000


def extract_text_from_url(url: str) -> tuple[str, bool]:
    """
    Returns (article_text, was_truncated).
    Tries trafilatura first (best quality), falls back to readability/bs4.
    """
    try:
        resp = requests.get(url, timeout=15, headers=REQUEST_HEADERS)
        resp.raise_for_status()
        raw_html = resp.text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch URL: {e}")

    article_text = ""

    # 1. trafilatura — handles JS-heavy sites much better
    if HAS_TRAFILATURA:
        extracted = trafilatura.extract(
            raw_html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if extracted and len(extracted.strip()) > 100:
            article_text = extracted.strip()

    # 2. readability + bs4 fallback
    if not article_text:
        try:
            if HAS_READABILITY:
                from readability import Document
                doc = Document(raw_html)
                html = doc.summary()
            else:
                html = raw_html
            soup = BeautifulSoup(html, "html.parser")
            article_text = soup.get_text(separator="\n", strip=True)
        except Exception as e:
            raise RuntimeError(f"Failed to extract article text: {e}")

    if not article_text or len(article_text.strip()) < 50:
        raise RuntimeError("Could not extract readable text from this URL. The page may require JavaScript or a login.")

    truncated = len(article_text) > MAX_ARTICLE_CHARS
    return article_text[:MAX_ARTICLE_CHARS], truncated


# ── FastAPI app + rate limiter ────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Provider functions ────────────────────────────────────────
def _analyze_with_ollama(article_text: str, lang: str) -> dict | None:
    if not ollama_client:
        return None
    try:
        response = ollama_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": get_system_prompt(lang)},
                {"role": "user", "content": f"Analyze this news article:\n\n---\n{article_text}\n---"},
            ],
            temperature=0.3,
        )
        return _parse_llm_response(response.choices[0].message.content)
    except Exception:
        return None


def _analyze_with_groq(article_text: str, lang: str) -> tuple[dict | None, str | None]:
    if not groq_client:
        return None, None
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": get_system_prompt(lang)},
                {"role": "user", "content": f"Analyze this news article:\n\n---\n{article_text}\n---"},
            ],
            temperature=0.3,
        )
        return _parse_llm_response(response.choices[0].message.content), None
    except Exception as e:
        return None, str(e)


def _analyze_with_gemini(article_text: str, lang: str) -> tuple[dict | None, str | None]:
    try:
        genai = _get_gemini_client()
    except RuntimeError as e:
        return None, str(e)
    if not genai:
        return None, None
    try:
        full_prompt = f"{get_system_prompt(lang)}\n\nAnalyze this news article:\n\n---\n{article_text}\n---"
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(full_prompt)
        return _parse_llm_response(response.text if response else None), None
    except Exception as e:
        return None, str(e)


def _analyze_sync(article_text: str, lang: str) -> tuple[dict, str]:
    """
    Runs provider chain synchronously (called via asyncio.to_thread).
    Returns (result_dict, provider_name_used).
    """
    result = _analyze_with_ollama(article_text, lang)
    if result:
        return result, "ollama"

    result, groq_err = _analyze_with_groq(article_text, lang)
    if result:
        return result, "groq"

    result, gemini_err = _analyze_with_gemini(article_text, lang)
    if result:
        return result, "gemini"

    errors = []
    if groq_err:
        errors.append(f"Groq: {groq_err}")
    if gemini_err:
        errors.append(f"Gemini: {gemini_err}")

    err_detail = (
        " | ".join(errors) if errors
        else (
            "No AI provider available. "
            "Get a free Groq API key at https://console.groq.com "
            "and add GROQ_API_KEY=your_key to backend/.env"
        )
    )
    raise HTTPException(status_code=503, detail=err_detail)


# ── Pydantic models ───────────────────────────────────────────
from pydantic import BaseModel
from typing import Optional


class AnalyzeNewsRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None


class SourceInfo(BaseModel):
    domain: str
    score: int
    label: str
    credibility: str


class AnalyzeNewsResponse(BaseModel):
    bias: Optional[str] = None
    manipulation_techniques: Optional[list] = None
    highlights: Optional[list] = None
    confidence: Optional[int] = None
    explanation: Optional[str] = None
    provider_used: Optional[str] = None
    detected_language: Optional[str] = None
    was_truncated: Optional[bool] = None
    source_info: Optional[SourceInfo] = None


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/test")
async def test_api():
    providers = []
    if ollama_client:
        providers.append("ollama (local, free)")
    if groq_client:
        providers.append("groq (free tier)")
    if GEMINI_API_KEY:
        providers.append("gemini (free tier)")
    return {
        "message": "API is running",
        "ai_providers": providers or ["none - set GROQ_API_KEY in backend/.env"],
        "trafilatura": HAS_TRAFILATURA,
        "langdetect": HAS_LANGDETECT,
    }


@app.post("/analyze-news", response_model=AnalyzeNewsResponse)
@limiter.limit("10/minute")
async def analyze_news(request: Request, body: AnalyzeNewsRequest):
    if not body.text and not body.url:
        return AnalyzeNewsResponse()

    article_text = ""
    was_truncated = False
    source_info_data = get_source_info(body.url)

    if body.text and body.text.strip():
        raw = body.text.strip()
        was_truncated = len(raw) > MAX_ARTICLE_CHARS
        article_text = raw[:MAX_ARTICLE_CHARS]
    elif body.url and body.url.strip():
        try:
            article_text, was_truncated = await asyncio.to_thread(
                extract_text_from_url, body.url.strip()
            )
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        return AnalyzeNewsResponse()

    if not article_text or len(article_text.strip()) < 50:
        return AnalyzeNewsResponse(
            explanation="Could not extract enough text from the article to analyze.",
            manipulation_techniques=[],
            highlights=[],
        )

    # Detect language for prompt selection
    lang = await asyncio.to_thread(detect_language, article_text)

    try:
        # Run blocking provider chain in thread pool
        result, provider_used = await asyncio.to_thread(_analyze_sync, article_text, lang)

        def to_str_list(val):
            if not val:
                return []
            return [str(x) for x in val] if isinstance(val, (list, tuple)) else [str(val)]

        # Parse confidence (model may return string or int)
        confidence = None
        raw_conf = result.get("confidence")
        if raw_conf is not None:
            try:
                confidence = max(0, min(100, int(raw_conf)))
            except (ValueError, TypeError):
                confidence = None

        source_info = SourceInfo(**source_info_data) if source_info_data else None

        return AnalyzeNewsResponse(
            bias=str(result["bias"]) if result.get("bias") else None,
            manipulation_techniques=to_str_list(result.get("manipulation_techniques")),
            highlights=to_str_list(result.get("highlights")),
            confidence=confidence,
            explanation=str(result["explanation"]) if result.get("explanation") else None,
            provider_used=provider_used,
            detected_language=lang,
            was_truncated=was_truncated,
            source_info=source_info,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "rate limit" in err_msg.lower() or "quota" in err_msg.lower():
            raise HTTPException(status_code=503, detail="API rate limit aşıldı. Lütfen 1 dakika bekleyip tekrar deneyin.")
        if "401" in err_msg or "invalid" in err_msg.lower() or "api_key" in err_msg.lower():
            raise HTTPException(status_code=503, detail="Geçersiz API key. Lütfen backend/.env dosyasını kontrol edin.")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {err_msg}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)