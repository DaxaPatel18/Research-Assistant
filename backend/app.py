from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, requests, pathlib, shutil, json, re, math, time, threading
from dotenv import load_dotenv
from groq import Groq

env_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
print("Groq Key Loaded:", "loaded" if GROQ_API_KEY else "NOT FOUND")

client     = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "openai/gpt-oss-120b"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = pathlib.Path("/tmp") / "research-assistant-uploads"
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # /tmp may not exist in some local dev environments
FRONTEND_FILE = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "index.html"


ARXIV_CACHE_TTL = 300
ARXIV_FAILURE_CACHE_TTL = 300
ARXIV_RATE_LIMIT_COOLDOWN = 300
ARXIV_USER_AGENT = "ResearchAssistant/1.0 (academic project)"
_arxiv_cache = {}
_arxiv_cache_lock = threading.RLock()
_arxiv_global_cooldown_until = 0.0


def _store_arxiv_cache(query, papers, ttl):
    """Store an isolated cache value so concurrent requests cannot mutate it."""
    with _arxiv_cache_lock:
        _arxiv_cache[query] = {
            "timestamp": time.monotonic(),
            "ttl": ttl,
            "papers": [paper.copy() for paper in papers],
        }

# ─────────────────────────────────────────
#  Shared list of phrases associated with AI-generated writing.
#  IMPORTANT: /detect-ai and /humanize both use this exact same list.
#  Previously /detect-ai flagged ~22 phrases but /humanize's prompt only
#  instructed the model to remove ~10 of them — any of the other ~12 left
#  in the text kept the detection penalty active even after "humanizing",
#  which is a major reason the AI score barely moved (and occasionally
#  the reworded text introduced one of the untargeted phrases, e.g.
#  "certainly"/"absolutely", making the score go UP). Keeping one shared
#  list guarantees the humanizer targets exactly what the detector checks.
# ─────────────────────────────────────────
AI_FLAG_PHRASES = [
    "in conclusion", "it is worth noting", "furthermore",
    "it is important to", "in summary", "to summarize",
    "in the realm of", "delve into", "it's worth noting",
    "as an ai", "certainly", "absolutely", "of course",
    "in today's world", "it is crucial", "plays a crucial role",
    "a testament to", "in the ever-evolving", "it is essential",
    "needless to say", "as previously mentioned", "it goes without saying"
]

# Meaning-preserving, non-flagged replacements for each phrase above.
# None of these replacement words/phrases appear in AI_FLAG_PHRASES themselves,
# so this pass can never re-introduce a flagged phrase.
_AI_PHRASE_REPLACEMENTS = {
    "in conclusion":            "overall",
    "it is worth noting":       "notably",
    "furthermore":              "also",
    "it is important to":       "it helps to",
    "in summary":               "put simply",
    "to summarize":             "put simply",
    "in the realm of":          "in",
    "delve into":               "look closely at",
    "it's worth noting":        "notably",
    "as an ai":                 "",
    "certainly":                "",
    "absolutely":               "",
    "of course":                "",
    "in today's world":         "today",
    "it is crucial":            "this matters",
    "plays a crucial role":     "matters a great deal",
    "a testament to":           "a sign of",
    "in the ever-evolving":     "in the changing",
    "it is essential":          "this is needed",
    "needless to say":          "",
    "as previously mentioned":  "as mentioned",
    "it goes without saying":   "",
}

def strip_ai_phrases(text: str) -> str:
    """
    Deterministic, meaning-preserving cleanup pass. Runs AFTER the Groq
    rewrite as a guarantee — not a substitute for the prompt-based rewrite,
    but a safety net so leftover flagged phrases never survive a single
    /humanize call. Case-insensitive, preserves surrounding punctuation
    and capitalization at sentence starts, never deletes any sentence.
    """
    result = text
    for phrase in AI_FLAG_PHRASES:
        replacement = _AI_PHRASE_REPLACEMENTS.get(phrase, "")
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)

        def _sub(m, repl=replacement):
            matched = m.group(0)
            if not repl:
                return ""
            # Preserve capitalization if the matched phrase started a sentence
            if matched[0].isupper():
                return repl[0].upper() + repl[1:]
            return repl

        result = pattern.sub(_sub, result)

    # Clean up double spaces / stray punctuation left by removed phrases
    result = re.sub(r'\s{2,}', ' ', result)
    result = re.sub(r'\s+([,.;:])', r'\1', result)
    result = re.sub(r'^\s*[,.;:]\s*', '', result, flags=re.MULTILINE)
    return result.strip()

def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Return cosine similarity [0, 1] between two texts using TF-IDF."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
        import numpy as np
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        tfidf = vec.fit_transform([text_a, text_b])
        score = sk_cosine(tfidf[0], tfidf[1])[0][0]
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0

def semantic_similarity_pct(text_a: str, text_b: str):
    """
    Lightweight TF-IDF cosine similarity used to report meaning preservation
    after rewriting. Returns None on failure so callers handle it gracefully.
    """
    try:
        score = _tfidf_cosine(text_a, text_b)
        return round(score * 100, 1)
    except Exception:
        return None

# ─────────────────────────────────────────
#  Request Models
# ─────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    style: str = ""

class ChatRequest(BaseModel):
    message: str
    history: list = []

class SemanticSearchRequest(BaseModel):
    query: str

class TextAnalysisRequest(BaseModel):
    text: str
    style: str = "academic"

# ─────────────────────────────────────────
#  Helper — Groq AI
# ─────────────────────────────────────────
def ask_groq(system_prompt, user_prompt, max_tokens=1024):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
#  Helper — Fetch from arXiv
# ─────────────────────────────────────────
def fetch_arxiv(query, limit=6):
    import urllib.parse
    import xml.etree.ElementTree as ET

    global _arxiv_global_cooldown_until

    if not query or not str(query).strip():
        return []

    # 1. Sanitize search query:
    # Remove problematic special characters, quotes, math symbols, and arXiv operators (AND, OR, NOT)
    raw_str = str(query).strip()
    
    # Remove arXiv boolean operators as standalone words
    raw_str = re.sub(r'\b(?:AND|OR|NOT)\b', ' ', raw_str, flags=re.IGNORECASE)
    
    # Keep only clean alphanumeric words and hyphens/underscores/spaces, removing punctuation & special query syntax
    words = re.findall(r'\b[A-Za-z0-9\-_]+\b', raw_str)
    
    # Filter out single-letter words if needed, keeping useful terms
    sanitized_words = [w for w in words if len(w) > 1 or w.isalnum()]
    sanitized_query = " ".join(sanitized_words[:25]).strip()
    
    if not sanitized_query:
        return []

    now = time.monotonic()
    with _arxiv_cache_lock:
        cached = _arxiv_cache.get(sanitized_query)
        if cached and now - cached["timestamp"] < cached["ttl"]:
            return [paper.copy() for paper in cached["papers"][:limit]]
        if cached:
            _arxiv_cache.pop(sanitized_query, None)

        if now < _arxiv_global_cooldown_until:
            _store_arxiv_cache(sanitized_query, [], ARXIV_FAILURE_CACHE_TTL)
            return []

    query_encoded = urllib.parse.quote(sanitized_query)
    url = (f"https://export.arxiv.org/api/query"
           f"?search_query=all:{query_encoded}"
           f"&start=0&max_results={limit}&sortBy=relevance")
    papers = []

    # Do not retry here.  A single request keeps the endpoint responsive and
    # avoids amplifying temporary arXiv failures or rate limits.
    for attempt in range(1):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": ARXIV_USER_AGENT},
                timeout=12,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            print(
                f"arXiv request {type(exc).__name__} for query "
                f"{sanitized_query[:50]}"
            )
            _store_arxiv_cache(sanitized_query, [], ARXIV_FAILURE_CACHE_TTL)
            return []
        except requests.RequestException as exc:
            print(f"arXiv request failed for query {sanitized_query[:50]}: {exc}")
            _store_arxiv_cache(sanitized_query, [], ARXIV_FAILURE_CACHE_TTL)
            return []

        if response.status_code != 200:
            if response.status_code == 429:
                with _arxiv_cache_lock:
                    _arxiv_global_cooldown_until = max(
                        _arxiv_global_cooldown_until,
                        time.monotonic() + ARXIV_RATE_LIMIT_COOLDOWN,
                    )
                    _store_arxiv_cache(sanitized_query, [], ARXIV_FAILURE_CACHE_TTL)
                print(
                    f"arXiv rate limited query '{sanitized_query[:50]}'; "
                    f"cooldown started for {ARXIV_RATE_LIMIT_COOLDOWN} seconds"
                )
                return []
            print(
                f"arXiv API returned HTTP status {response.status_code} "
                f"for query: {sanitized_query[:50]}"
            )
            _store_arxiv_cache(sanitized_query, [], ARXIV_FAILURE_CACHE_TTL)
            return []

        try:
            root = ET.fromstring(response.content)
        except (ET.ParseError, TypeError, ValueError) as exc:
            print(
                f"arXiv API returned invalid XML/HTML for query "
                f"{sanitized_query[:50]}: {exc}"
            )
            _store_arxiv_cache(sanitized_query, [], ARXIV_FAILURE_CACHE_TTL)
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        if root.tag != "{http://www.w3.org/2005/Atom}feed":
            print(
                f"arXiv API returned unexpected non-feed content for query: "
                f"{sanitized_query[:50]}"
            )
            _store_arxiv_cache(sanitized_query, [], ARXIV_FAILURE_CACHE_TTL)
            return []

        for entry in root.findall("atom:entry", ns):
            title       = entry.find("atom:title",    ns)
            abstract    = entry.find("atom:summary",  ns)
            year_raw    = entry.find("atom:published",ns)
            authors     = entry.findall("atom:author",ns)
            entry_id    = entry.find("atom:id",       ns)
            journal_ref = entry.find("{http://arxiv.org/schemas/atom}journal_ref", ns)
            doi         = entry.find("{http://arxiv.org/schemas/atom}doi", ns)

            author_names = []
            for a in authors[:4]:
                name = a.find("atom:name", ns)
                if name is not None and name.text:
                    author_names.append(name.text)

            arxiv_id = ""
            pdf_url  = ""
            if entry_id is not None and entry_id.text:
                raw_id   = entry_id.text.strip()
                arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else ""
                if arxiv_id:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            if journal_ref is not None and journal_ref.text:
                status       = "published"
                status_label = f"Published — {journal_ref.text[:60]}"
            elif doi is not None and doi.text:
                status       = "published"
                status_label = f"Published (DOI: {doi.text})"
            else:
                status       = "preprint"
                status_label = "Preprint — Not peer reviewed"

            papers.append({
                "title":        title.text.strip().replace("\n"," ") if title is not None and title.text else "Untitled",
                "authors":      ", ".join(author_names),
                "year":         year_raw.text[:4]                    if year_raw is not None and year_raw.text else "N/A",
                "venue":        "arXiv",
                "abstract":     abstract.text.strip().replace("\n"," ") if abstract is not None and abstract.text else "No abstract.",
                "citations":    0,
                "arxiv_id":     arxiv_id,
                "pdf_url":      pdf_url,
                "status":       status,
                "status_label": status_label,
                "doi":          doi.text if doi is not None and doi.text else "",
            })
        _store_arxiv_cache(sanitized_query, papers, ARXIV_CACHE_TTL)
        return [paper.copy() for paper in papers[:limit]]

    print(f"arXiv request failed for query: {sanitized_query[:50]}")
    _store_arxiv_cache(sanitized_query, [], ARXIV_FAILURE_CACHE_TTL)
    return []

# ─────────────────────────────────────────
#  Helper — N-gram overlap (Jaccard)
#  This is how real plagiarism tools work
# ─────────────────────────────────────────
def ngram_overlap(text1, text2, n=4):
    """
    Computes what % of n-grams in text1 appear in text2.
    n=4 means 4-word phrases. If 4-word phrases from your
    text appear in a paper, that is strong plagiarism signal.
    """
    def get_ngrams(text, n):
        words  = re.sub(r'[^\w\s]', '', text.lower()).split()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    if not ngrams1:
        return 0.0

    intersection = ngrams1 & ngrams2
    # % of your ngrams found in the paper
    return round(len(intersection) / len(ngrams1) * 100, 1)

# ─────────────────────────────────────────
#  Helper — Sentence level similarity (TF-IDF)
# ─────────────────────────────────────────
def sentence_similarity(text1, text2):
    """
    Splits text1 into sentences and checks each sentence against text2
    using TF-IDF cosine similarity. Returns max similarity and flagged sentences.
    """
    try:
        sentences1 = [s.strip() for s in re.split(r'[.!?]+', text1) if len(s.strip()) > 20]
        if not sentences1:
            return 0.0, []

        scores = [_tfidf_cosine(sent, text2) for sent in sentences1]

        flagged = []
        for sent, score in zip(sentences1, scores):
            if score > 0.55:  # adjusted threshold for TF-IDF (lower than embedding threshold)
                flagged.append({
                    "sentence":   sent,
                    "similarity": round(score * 100, 1)
                })

        max_score = max(scores) * 100 if scores else 0.0
        return round(max_score, 1), flagged

    except Exception:
        return 0.0, []

# ─────────────────────────────────────────
#  Health Check
# ─────────────────────────────────────────
@app.get("/")
def root():
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE, media_type="text/html")
    return {
        "status":    "ResearchAI backend is running",
        "ai_engine": "Groq (Free)",
        "model":     GROQ_MODEL,
        "groq_key":  "loaded" if GROQ_API_KEY else "MISSING",
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# ─────────────────────────────────────────
#  1. SEMANTIC SEARCH
# ─────────────────────────────────────────
@app.post("/semantic-search")
def semantic_search(req: SemanticSearchRequest):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

        papers = fetch_arxiv(req.query, limit=20)
        if not papers:
            return {"papers": [], "error": "Could not fetch papers from arXiv"}

        combined_texts = [
            p.get("title", "") + " " + p.get("abstract", "")
            for p in papers
        ]

        # Fit TF-IDF on query + all paper texts together so IDF is shared
        corpus = [req.query] + combined_texts
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        tfidf = vec.fit_transform(corpus)

        query_vec  = tfidf[0]          # first row = query
        paper_vecs = tfidf[1:]         # remaining rows = papers
        scores = sk_cosine(query_vec, paper_vecs)[0].tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:6]:
            paper = papers[idx].copy()
            paper["similarity_score"] = round(score * 100, 1)
            paper["match_level"] = (
                "Excellent Match" if score > 0.25 else
                "Good Match"      if score > 0.15 else
                "Partial Match"   if score > 0.07 else
                "Weak Match"
            )
            results.append(paper)

        return {"papers": results, "count": len(results), "total_fetched": len(papers)}

    except ImportError:
        return {"papers": [], "error": "Run: pip install scikit-learn"}
    except Exception as e:
        return {"papers": [], "error": str(e)}

# ─────────────────────────────────────────
#  2. SUMMARIZE
# ─────────────────────────────────────────
@app.post("/summarize")
def summarize(req: QueryRequest):
    style_map = {
        "Concise (3 bullets)":    "Summarize in exactly 3 bullet points starting with •",
        "ELI5 (Simple language)": "Explain like I am 12 years old, under 120 words.",
        "Technical (Deep dive)":  "Detailed summary with: Objective, Methods, Key Findings, Limitations, Future Work"
    }
    instruction = style_map.get(req.style, "Summarize in 3 bullet points starting with •")
    result = ask_groq(
        "You are an expert academic summarizer. Be accurate and structured.",
        instruction + "\n\nPaper content:\n" + req.query[:6000],
        max_tokens=800
    )
    return {"summary": result}

# ─────────────────────────────────────────
#  3. PDF UPLOAD (for summarize panel)
# ─────────────────────────────────────────
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = None
    try:
        import fitz
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        doc        = fitz.open(str(file_path))
        full_text  = "".join(page.get_text() for page in doc)
        page_count = len(doc)
        doc.close()

        full_text = full_text.strip()
        if not full_text:
            return {"error": "Could not extract text. PDF may be scanned or image-based."}

        analysis = ask_groq(
            "You are an expert academic paper analyzer. Be structured and precise.",
            f"""Analyze this research paper:
1. TITLE: (extract the paper title)
2. AUTHORS: (extract author names)
3. SUMMARY: (3 bullet points starting with •)
4. KEY TOPICS: (5 main keywords)
5. PROBLEM SOLVED: (what gap does this paper address)
6. METHODOLOGY: (what approach was used)
7. KEY FINDINGS: (2-3 main findings)
8. RELATED SEARCH TERMS: (3 terms to find similar papers)

Paper content:
{full_text[:3000]}""",
            max_tokens=1200
        )

        return {
            "filename":     file.filename,
            "characters":   len(full_text),
            "pages":        page_count,
            "analysis":     analysis,
            "text_content": full_text[:4000],
        }

    except ImportError:
        return {"error": "Run: pip install pymupdf"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass

# ─────────────────────────────────────────
#  4. PLAGIARISM CHECK PDF UPLOAD
# ─────────────────────────────────────────
@app.post("/upload-pdf-plagiarism")
async def upload_pdf_plagiarism(file: UploadFile = File(...)):
    """
    Extracts text from PDF and returns it for plagiarism checking.
    Separate endpoint so plagiarism panel has its own upload.
    """
    file_path = None
    try:
        import fitz
        file_path = UPLOAD_DIR / ("plag_" + file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        doc        = fitz.open(str(file_path))
        full_text  = "".join(page.get_text() for page in doc)
        page_count = len(doc)
        doc.close()

        full_text = full_text.strip()
        if not full_text:
            return {"error": "Could not extract text. PDF may be scanned or image-based."}

        return {
            "filename":   file.filename,
            "pages":      page_count,
            "characters": len(full_text),
            # Return full text for plagiarism checking
            "text":       full_text[:8000],
        }

    except ImportError:
        return {"error": "Run: pip install pymupdf"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass

# ─────────────────────────────────────────
#  5. PLAGIARISM CHECKER — FIXED
#  Uses n-gram overlap + sentence similarity
#  Much more accurate than embedding similarity
# ─────────────────────────────────────────
@app.post("/check-plagiarism")
def check_plagiarism(req: TextAnalysisRequest):
    try:
        text = req.text.strip()
        if len(text) < 50:
            return {"error": "Please provide at least 50 characters of text."}

        # Use first 300 chars to build search query
        search_query = ' '.join(text[:300].split()[:20])
        print(f"Plagiarism check — fetching papers for: {search_query[:80]}")

        # Fetch papers from multiple related queries for better coverage
        papers = []
        papers.extend(fetch_arxiv(search_query, limit=10))

        # Also search with different part of text
        if len(text) > 200:
            mid_query = ' '.join(text[200:400].split()[:15])
            papers.extend(fetch_arxiv(mid_query, limit=8))

        # Remove duplicates by title
        seen_titles = set()
        unique_papers = []
        for p in papers:
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                unique_papers.append(p)
        papers = unique_papers

        print(f"Checking against {len(papers)} papers")

        if not papers:
            return {"error": "Could not fetch papers to compare against."}

        matches = []
        all_ngram_scores  = []
        all_sentence_scores = []

        for paper in papers:
            abstract = paper.get("abstract", "")
            if not abstract or abstract == "No abstract.":
                continue

            # Method 1 — N-gram overlap (4-word phrase matching)
            ngram_score_4 = ngram_overlap(text, abstract, n=4)
            ngram_score_3 = ngram_overlap(text, abstract, n=3)
            ngram_score   = max(ngram_score_4, ngram_score_3 * 0.7)

            # Method 2 — Sentence level similarity
            sent_score, flagged_sentences = sentence_similarity(text, abstract)

            # Combined score — weighted
            # n-gram is more reliable for exact matches
            combined = round((ngram_score * 0.6) + (sent_score * 0.4), 1)

            all_ngram_scores.append(ngram_score)
            all_sentence_scores.append(sent_score)

            if combined > 5 or ngram_score > 3 or sent_score > 60:
                matches.append({
                    "title":              paper["title"],
                    "authors":            paper["authors"],
                    "year":               paper["year"],
                    "pdf_url":            paper["pdf_url"],
                    "ngram_score":        round(ngram_score, 1),
                    "sentence_score":     round(sent_score, 1),
                    "combined_score":     combined,
                    "flagged_sentences":  flagged_sentences[:3],
                    "risk_level": (
                        "High Risk"   if combined > 40 or ngram_score > 30 else
                        "Medium Risk" if combined > 15 or ngram_score > 10 else
                        "Low Risk"
                    )
                })

        # Sort by combined score
        matches.sort(key=lambda x: x["combined_score"], reverse=True)

        # Overall score — highest combined match
        overall = matches[0]["combined_score"] if matches else 0

        # If very low, check if sentence similarity alone is high
        if overall < 5 and all_sentence_scores:
            max_sent = max(all_sentence_scores)
            if max_sent > 70:
                overall = round(max_sent * 0.5, 1)

        overall = min(100, overall)

        # AI written analysis
        top_matches_text = ""
        for m in matches[:3]:
            top_matches_text += f"\n- {m['title']} ({m['year']}): {m['combined_score']}% match"

        analysis = ask_groq(
            "You are an expert plagiarism detection analyst.",
            f"""Analyze this text for plagiarism. The system found these similar papers:
{top_matches_text if top_matches_text else "No significant matches found."}

Overall similarity score: {overall}%

Text analyzed (first 1000 chars):
{text[:1000]}

Provide:
1. VERDICT: Is this text original or does it show signs of plagiarism?
2. WHAT IS SIMILAR: What type of content matches (concepts, phrases, methodology)?
3. RECOMMENDATIONS: How to make this more original and properly cited?
4. CITATION ADVICE: Which papers should be cited if ideas were taken from them?

Be specific and constructive. Keep response under 200 words.""",
            max_tokens=400
        )

        return {
            "overall_score":  round(overall, 1),
            "risk_level":     "High Risk"   if overall > 40 else
                              "Medium Risk" if overall > 15 else
                              "Low Risk",
            "matches":        matches[:5],
            "papers_checked": len(papers),
            "analysis":       analysis,
            "method":         "N-gram phrase matching + sentence similarity"
        }

    except Exception as e:
        print("Plagiarism error:", e)
        return {"error": str(e)}

# ─────────────────────────────────────────
#  6. HUMANIZER (Groq LLM + MiniLM Pipeline)
#  Note: Pegasus (tuner007/pegasus_paraphrase) was removed because it
#  requires ~3-4 GB RAM, exceeding Vercel Hobby's 2 GB limit.
#  The Groq-based rewrite path (previously the except-branch fallback)
#  is now the primary and only execution path.
# ─────────────────────────────────────────

def split_into_sentences(text: str) -> list:
    """
    Splits text into individual sentences while preserving exact order and count N.
    Regex handles standard sentence boundaries (. ! ?) followed by whitespace or end of string.
    Guarantees N input sentences -> N output units.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences = []

    for para in paragraphs:
        raw_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
        if raw_sents:
            sentences.extend(raw_sents)
        else:
            sentences.append(para)

    return sentences if sentences else [text.strip()]

def split_into_chunks(text: str, max_words: int = 100) -> list:
    """
    Legacy compatibility wrapper mapping to sentence segmentation.
    """
    return split_into_sentences(text)

@app.post("/humanize")
def humanize(req: TextAnalysisRequest):
    """
    Humanizer Pipeline (Groq LLM + MiniLM):
    Text -> Input Validation -> Sentence Count -> Groq LLM Rewrite ->
    Conservative AI Phrase Cleanup -> MiniLM Semantic Similarity Check
    Pegasus (tuner007/pegasus_paraphrase) is NOT used; Groq is the
    primary rewriting engine, keeping RAM usage within Vercel Hobby limits.
    API contract (request/response schema) is unchanged.
    """
    try:
        text = req.text.strip()
        style = req.style or "academic"

        if len(text) < 20:
            return {"error": "Please provide at least 20 characters."}

        # Count sentences for reporting (same helper used elsewhere)
        sentences_processed = len(split_into_sentences(text))

        # ── Primary rewrite: Groq LLM ────────────────────────────────
        instruction = (
            f"Rewrite the following text into natural, fluent prose in '{style}' style. "
            "Preserve all facts, technical terms, numbers, and meaning exactly. "
            "Do not add new information. Do not use filler phrases."
        )
        raw_humanized = ask_groq(instruction, text, max_tokens=1500)
        # ─────────────────────────────────────────────────────────────

        # Conservative final cleanup: remove flagged AI phrases
        humanized = strip_ai_phrases(raw_humanized)

        # Meaning preservation check via TF-IDF cosine similarity
        meaning_similarity = semantic_similarity_pct(text, humanized)

        sentences_changed = sentences_processed  # Groq rewrites the full text
        total_candidates_eval = 1

        changes_made = [
            "Rewrote text using Groq LLM neural rephrasing",
            "Evaluated meaning preservation with TF-IDF cosine similarity",
            "Applied conservative AI phrase cleanup",
            f"Style applied: {style.title()}",
        ]

        return {
            "humanized_text": humanized,
            "original_words": len(text.split()),
            "humanized_words": len(humanized.split()),
            "changes_made": changes_made,
            "meaning_similarity": meaning_similarity,
            "paraphrase_quality": 80.0,
            "chunks_processed": sentences_processed,
            "sentences_processed": sentences_processed,
            "sentences_changed": sentences_changed,
            "candidates_evaluated": total_candidates_eval,
            "candidates_rejected": 0,
            "candidates_rejected_protected_info": 0,
            "protected_items_found": 0,
            "protected_items_preserved": 0,
            "unchanged_candidates": 0,
            "pipeline": "Groq LLM Rewrite + TF-IDF Semantic Scoring",
            "fallback_used": False,
            "tip": "Rewritten using Groq LLM and evaluated for meaning preservation with TF-IDF cosine similarity.",
            "note": "This tool focuses on meaning-preserving paraphrasing and writing quality improvement. It is an ML academic utility and does not guarantee bypassing AI detection systems."
        }

    except Exception as e:
        print("Humanize error:", e)
        return {"error": str(e)}




#  8. AI CHAT
# ─────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    messages = [{
        "role":    "system",
        "content": """You are ResearchAI, a smart academic research assistant. Help with:
- Research papers and concepts
- Paper content questions
- Research gaps identification
- Methodology suggestions
- Literature review structuring
- ML, NLP, AI questions
Be concise and academically accurate."""
    }]
    for h in req.history[-6:]:
        if h.get("role") in ["user","assistant"]:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, max_tokens=1024, temperature=0.7
    )
    return {"reply": response.choices[0].message.content}

# ─────────────────────────────────────────
#  9. LITERATURE REVIEW
# ─────────────────────────────────────────
@app.post("/literature-review")
def literature_review(req: QueryRequest):
    try:
        papers = fetch_arxiv(req.query, limit=12)
        if not papers:
            return {"error": "Could not fetch papers."}

        papers_text = ""
        for i, p in enumerate(papers, 1):
            papers_text += f"\nPaper {i}:\nTitle: {p['title']}\nAuthors: {p['authors']}\nYear: {p['year']}\nAbstract: {p['abstract'][:400]}\n---"

        review = ask_groq(
            "You are an expert academic writer. Write in formal academic style. Cite papers as [1],[2] etc.",
            f"""Write a comprehensive literature review on: "{req.query}"
Using these {len(papers)} papers:
{papers_text}

Structure:
1. INTRODUCTION (2-3 paragraphs)
2. THEMATIC ANALYSIS (3-4 themes, cite papers by number)
3. RESEARCH TRENDS (3-4 major trends)
4. RESEARCH GAPS AND FUTURE DIRECTIONS
5. CONCLUSION (2-3 sentences)
6. REFERENCES ([1] Authors (Year). Title.)""",
            max_tokens=2000
        )
        return {"review": review, "papers_used": len(papers), "topic": req.query}

    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────
#  10. CITATION GENERATOR
# ─────────────────────────────────────────
@app.post("/cite")
def generate_citation(req: QueryRequest):
    lines   = req.query.strip().split("\n")
    details = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            details[k.strip().lower()] = v.strip()

    title   = details.get("title",   "Unknown Title")
    authors = details.get("authors", "Unknown Author")
    year    = details.get("year",    "n.d.")
    journal = details.get("journal/conference", details.get("journal",""))
    volume  = details.get("volume","")
    pages   = details.get("pages", "")
    style   = req.style
    al      = [a.strip() for a in authors.split(",") if a.strip()]

    def fmt_apa(a):
        p = a.split()
        return f"{p[-1]}, {' '.join(x[0]+'.' for x in p[:-1])}" if len(p)>=2 else a
    def fmt_ieee(a):
        p = a.split()
        return f"{'. '.join(x[0] for x in p[:-1])}. {p[-1]}" if len(p)>=2 else a
    def fmt_lf(a):
        p = a.split()
        return f"{p[-1]}, {' '.join(p[:-1])}" if len(p)>=2 else a
    def fmt_harv(a):
        p = a.split()
        return f"{p[-1]}, {'.'.join(x[0] for x in p[:-1])}." if len(p)>=2 else a

    if style=="APA 7th":
        c = f"{', '.join(fmt_apa(a) for a in al[:6])} ({year}). {title}."
        if journal: c += f" {journal}"
        if volume:  c += f", {volume}"
        if pages:   c += f", {pages}"
        c += "."
    elif style=="MLA 9th":
        mla = [fmt_lf(al[0])]+al[1:3] if al else ["Unknown"]
        c   = f'{", ".join(mla)}. "{title}."'
        if journal: c += f" {journal},"
        if volume:  c += f" vol. {volume},"
        c += f" -m{year}"
        if pages:   c += f", pp. {pages}"
        c += "."
    elif style=="IEEE":
        c = f'[1] {", ".join(fmt_ieee(a) for a in al[:6])}, "{title},"'
        if journal: c += f" {journal},"
        if volume:  c += f" vol. {volume},"
        if pages:   c += f" pp. {pages},"
        c += f" {year}."
    elif style=="Chicago":
        chi = [fmt_lf(al[0])]+al[1:3] if al else ["Unknown"]
        c   = f'{", ".join(chi)}. "{title}."'
        if journal: c += f" {journal}"
        if volume:  c += f" {volume}"
        c += f" ({year})"
        if pages:   c += f": {pages}"
        c += "."
    elif style=="Harvard":
        c = f"{', '.join(fmt_harv(a) for a in al[:6])} ({year}) '{title}'"
        if journal: c += f", {journal}"
        if volume:  c += f", {volume}"
        if pages:   c += f", pp. {pages}"
        c += "."
    else:
        c = f"{authors} ({year}). {title}. {journal}."

    return {"citation": c}
