from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, requests, pathlib, shutil, json, re, math
from dotenv import load_dotenv
from groq import Groq

env_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
print("Groq Key Loaded:", repr(GROQ_API_KEY[:15]) if GROQ_API_KEY else "NOT FOUND")

client     = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = pathlib.Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

semantic_model = None

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

def semantic_similarity_pct(text_a: str, text_b: str):
    """
    Uses the existing all-MiniLM-L6-v2 model to report how much of the
    original meaning was preserved after rewriting. Returns None if the
    NLP model can't be loaded, so callers must handle that gracefully.
    """
    try:
        from sentence_transformers import util
        model = get_semantic_model()
        emb = model.encode([text_a, text_b], convert_to_tensor=True)
        score = util.cos_sim(emb[0], emb[1]).item()
        return round(max(0.0, min(1.0, score)) * 100, 1)
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
#  Helper — Load NLP Model
# ─────────────────────────────────────────
def get_semantic_model():
    global semantic_model
    if semantic_model is None:
        print("Loading NLP model...")
        from sentence_transformers import SentenceTransformer
        semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("NLP model ready!")
    return semantic_model

# ─────────────────────────────────────────
#  Helper — Fetch from arXiv
# ─────────────────────────────────────────
def fetch_arxiv(query, limit=6):
    import urllib.parse
    import xml.etree.ElementTree as ET

    query_encoded = urllib.parse.quote(query)
    url = (f"https://export.arxiv.org/api/query"
           f"?search_query=all:{query_encoded}"
           f"&start=0&max_results={limit}&sortBy=relevance")
    r    = requests.get(url, timeout=20)
    root = ET.fromstring(r.content)
    ns   = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
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
            if name is not None:
                author_names.append(name.text)

        arxiv_id = ""
        pdf_url  = ""
        if entry_id is not None:
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
            "title":        title.text.strip().replace("\n"," ") if title    is not None else "Untitled",
            "authors":      ", ".join(author_names),
            "year":         year_raw.text[:4]                    if year_raw is not None else "N/A",
            "venue":        "arXiv",
            "abstract":     abstract.text.strip().replace("\n"," ") if abstract is not None else "No abstract.",
            "citations":    0,
            "arxiv_id":     arxiv_id,
            "pdf_url":      pdf_url,
            "status":       status,
            "status_label": status_label,
            "doi":          doi.text if doi is not None else "",
        })
    return papers

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
#  Helper — Sentence level similarity
# ─────────────────────────────────────────
def sentence_similarity(text1, text2):
    """
    Splits text into sentences and checks each sentence
    against the paper abstract for similarity.
    Returns max similarity found and flagged sentences.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model = get_semantic_model()

        sentences1 = [s.strip() for s in re.split(r'[.!?]+', text1) if len(s.strip()) > 20]
        if not sentences1:
            return 0.0, []

        # Encode all sentences and target text
        emb1 = model.encode(sentences1, convert_to_tensor=True)
        emb2 = model.encode([text2],    convert_to_tensor=True)

        scores = util.cos_sim(emb1, emb2)[:,0].tolist()

        flagged = []
        for sent, score in zip(sentences1, scores):
            if score > 0.75:  # high similarity threshold
                flagged.append({
                    "sentence":   sent,
                    "similarity": round(score * 100, 1)
                })

        max_score = max(scores) * 100 if scores else 0
        return round(max_score, 1), flagged

    except Exception:
        return 0.0, []

# ─────────────────────────────────────────
#  Health Check
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":    "ResearchAI backend is running",
        "ai_engine": "Groq (Free)",
        "model":     GROQ_MODEL,
        "groq_key":  "loaded" if GROQ_API_KEY else "MISSING",
    }

# ─────────────────────────────────────────
#  1. SEMANTIC SEARCH
# ─────────────────────────────────────────
@app.post("/semantic-search")
def semantic_search(req: SemanticSearchRequest):
    try:
        from sentence_transformers import util

        papers = fetch_arxiv(req.query, limit=20)
        if not papers:
            return {"papers": [], "error": "Could not fetch papers from arXiv"}

        model            = get_semantic_model()
        query_embedding  = model.encode(req.query, convert_to_tensor=True)
        combined_texts   = [p.get("title","")+" "+p.get("abstract","") for p in papers]
        paper_embeddings = model.encode(combined_texts, convert_to_tensor=True)
        scores           = util.cos_sim(query_embedding, paper_embeddings)[0].tolist()
        ranked           = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:6]:
            paper = papers[idx].copy()
            paper["similarity_score"] = round(score * 100, 1)
            paper["match_level"] = (
                "Excellent Match" if score > 0.7 else
                "Good Match"      if score > 0.5 else
                "Partial Match"   if score > 0.3 else
                "Weak Match"
            )
            results.append(paper)

        return {"papers": results, "count": len(results), "total_fetched": len(papers)}

    except ImportError:
        return {"papers": [], "error": "Run: pip install sentence-transformers torch"}
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

# ─────────────────────────────────────────
#  4. PLAGIARISM CHECK PDF UPLOAD
# ─────────────────────────────────────────
@app.post("/upload-pdf-plagiarism")
async def upload_pdf_plagiarism(file: UploadFile = File(...)):
    """
    Extracts text from PDF and returns it for plagiarism checking.
    Separate endpoint so plagiarism panel has its own upload.
    """
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
#  6. AI DETECTION
# ─────────────────────────────────────────
@app.post("/detect-ai")
def detect_ai(req: TextAnalysisRequest):
    try:
        text = req.text.strip()
        if len(text) < 100:
            return {"error": "Please provide at least 100 characters."}

        sentences    = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words        = text.split()
        avg_sent_len = len(words) / max(len(sentences), 1)

        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            mean_len  = sum(sent_lengths) / len(sent_lengths)
            variance  = sum((l - mean_len)**2 for l in sent_lengths) / len(sent_lengths)
            std_dev   = variance ** 0.5
        else:
            std_dev = 0

        text_lower      = text.lower()
        ai_phrase_count = sum(1 for phrase in AI_FLAG_PHRASES if phrase in text_lower)

        ai_analysis = ask_groq(
            "You are an expert AI text detection system. Analyze carefully.",
            f"""Analyze if this text was written by AI or human.

Text:
{text[:2000]}

Reply in EXACTLY this format:
AI_SCORE: [0-100]
CONFIDENCE: [Low/Medium/High]
VERDICT: [AI Generated / Likely AI / Uncertain / Likely Human / Human Written]

KEY INDICATORS:
- [indicator 1]
- [indicator 2]
- [indicator 3]

EXPLANATION:
[2-3 sentences]

HUMAN PATTERNS MISSING:
[What is absent]""",
            max_tokens=500
        )

        score_match = re.search(r'AI_SCORE:\s*(\d+)', ai_analysis)
        ai_score    = int(score_match.group(1)) if score_match else 50

        if std_dev < 3 and len(sentences) > 3:
            ai_score = min(100, ai_score + 10)
        if ai_phrase_count > 2:
            ai_score = min(100, ai_score + 4 * ai_phrase_count)

        ai_score = min(100, max(0, ai_score))

        return {
            "ai_score": ai_score,
            "verdict":  (
                "Almost Certainly AI" if ai_score > 85 else
                "Likely AI Generated" if ai_score > 65 else
                "Possibly AI"         if ai_score > 45 else
                "Likely Human"        if ai_score > 25 else
                "Almost Certainly Human"
            ),
            "risk_level": "High" if ai_score > 65 else "Medium" if ai_score > 40 else "Low",
            "stats": {
                "word_count":        len(words),
                "sentence_count":    len(sentences),
                "avg_sentence_len":  round(avg_sent_len, 1),
                "sentence_variance": round(std_dev, 1),
                "ai_phrases_found":  ai_phrase_count,
            },
            "analysis": ai_analysis
        }

    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────
#  7. HUMANIZER
# ─────────────────────────────────────────
@app.post("/humanize")
def humanize(req: TextAnalysisRequest):
    """
    Natural Academic Rewrite.

    Rewrites text so it reads naturally while preserving facts, technical
    terms, citations, numbers and claims. This is NOT a guaranteed AI-
    detector bypass — see the disclaimer in the response.

    Fix notes (why the previous version barely moved the AI score, and
    sometimes made it worse):
      1. /detect-ai penalizes ~22 specific phrases, but the old prompts
         only asked the model to avoid ~10 of them. Any of the other ~12
         left untouched kept the penalty active. Worse, a couple of styles
         could introduce "certainly"/"absolutely" as natural-sounding
         filler, which are ALSO on the flagged list — so the score could
         go up. Fixed by giving every style the exact same full list.
      2. Sentence-length variety was only a vague bullet point, so the
         model often produced text that was still fairly uniform. Fixed
         with concrete numeric guidance.
      3. There was no deterministic guarantee — a single LLM call is
         probabilistic and can simply miss instructions. Fixed by adding
         a rule-based cleanup pass (strip_ai_phrases) that runs after the
         rewrite and removes any flagged phrase that slipped through,
         without deleting or altering any factual content.
    """
    try:
        text  = req.text.strip()
        style = req.style or "academic"

        if len(text) < 50:
            return {"error": "Please provide at least 50 characters."}

        avoid_phrases_list = "\n".join(f'- "{p}"' for p in AI_FLAG_PHRASES)

        shared_rules = f"""
CRITICAL — do not use ANY of these phrases anywhere in the rewrite (they are
strong AI-writing signals). If the original text contains one, replace it
with a natural alternative instead of just deleting it:
{avoid_phrases_list}

CRITICAL — sentence rhythm: aim for a genuine mix of sentence lengths in
every paragraph — at least one shorter sentence (roughly 6-12 words) and at
least one longer sentence (roughly 22-30 words) per paragraph where the
content allows it. Uniform, evenly-paced sentences are a strong AI signal.

Preserve, exactly and without alteration:
- All technical terms, named methods, and jargon
- All citations and references (e.g., [1], (Smith, 2020))
- All numbers, statistics, percentages and dates
- All factual claims and the overall meaning
Do not invent new facts, examples, or claims that were not in the original.
Do not delete any factual content, only rephrase it.
"""

        style_instructions = {
            "academic": f"""You are an academic editor rewriting text so it reads like it was
written by a genuine researcher, while keeping a professional, academic tone.
{shared_rules}
Additional style guidance:
- Use contractions sparingly and only where natural for academic prose (don't, it's)
- Prefer plainer verbs over inflated ones where meaning is unaffected: "utilize"→"use", "demonstrate"→"show", "facilitate"→"help"
- Occasionally start a sentence with "And" or "But" if it reads naturally
- Vary paragraph rhythm; avoid restating the same sentence pattern twice in a row""",

            "casual": f"""You are rewriting text so it reads like a smart, casual student wrote it.
{shared_rules}
Additional style guidance:
- Use contractions freely (don't, it's, they're, we've, can't, won't)
- Natural informal connectors are fine: "basically", "the thing is", "what's interesting is"
- Add a little personal voice where appropriate: "I'd argue", "from what I can tell"
- Keep it conversational but still factually precise""",

            "natural": f"""You are rewriting text so it reads like a thoughtful person wrote it naturally.
{shared_rules}
Additional style guidance:
- Use occasional parenthetical asides or em dashes for a natural aside — they feel human
- Prefer concrete, specific phrasing over generic, uniform phrasing
- One-sentence paragraphs are fine occasionally
- Keep the tone genuine rather than performative"""
        }

        instruction = style_instructions.get(style, style_instructions["academic"])

        humanized_raw = ask_groq(
            instruction,
            f"""Rewrite the following text following all the rules above.
Do NOT add any explanation, preamble, or notes — output ONLY the rewritten text.

Original text:
{text}

Rewritten version:""",
            max_tokens=2000
        )

        # Deterministic safety net: guarantee no flagged phrase survives,
        # regardless of what the model actually did.
        humanized = strip_ai_phrases(humanized_raw)

        # ── Report what actually changed ──
        contractions = ["don't","it's","we've","they're","isn't","wasn't","can't","won't","that's"]
        contractions_added = [c for c in contractions if c in humanized.lower() and c not in text.lower()]

        phrases_removed = [p for p in AI_FLAG_PHRASES
                            if p in text.lower() and p not in humanized.lower()]

        changes_made = []
        if contractions_added:
            changes_made.append(f"Added contractions: {', '.join(contractions_added[:4])}")
        if phrases_removed:
            shown = phrases_removed[:4]
            more  = f" (+{len(phrases_removed)-4} more)" if len(phrases_removed) > 4 else ""
            changes_made.append(f"Removed AI-associated phrasing: {', '.join(shown)}{more}")

        orig_sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        hum_sents  = [s for s in re.split(r'[.!?]+', humanized) if s.strip()]
        if len(orig_sents) > 1 and len(hum_sents) > 1:
            orig_lens = [len(s.split()) for s in orig_sents]
            hum_lens  = [len(s.split()) for s in hum_sents]
            orig_var  = sum((l - sum(orig_lens)/len(orig_lens))**2 for l in orig_lens) / len(orig_lens)
            hum_var   = sum((l - sum(hum_lens)/len(hum_lens))**2 for l in hum_lens) / len(hum_lens)
            if hum_var > orig_var:
                changes_made.append("Increased sentence-length variety")

        if not changes_made:
            changes_made.append("Rewrote with more natural flow and academic voice")
        changes_made.append(f"Style applied: {style.title()}")

        # Honest meaning-preservation signal using the existing MiniLM model
        # (not a loop, not a detector bypass — just a diagnostic metric).
        meaning_similarity = semantic_similarity_pct(text, humanized)

        return {
            "humanized_text":     humanized,
            "original_words":     len(text.split()),
            "humanized_words":    len(humanized.split()),
            "changes_made":       changes_made,
            "meaning_similarity": meaning_similarity,  # 0-100, or null if unavailable
            "tip": "This is a natural academic rewrite, not a guaranteed AI-detector bypass. "
                   "You can re-run AI Detection on the result to see how it scores, but "
                   "detection scores vary between systems and are not definitive proof of authorship.",
            "note": "AI-detection tools (including the one in this app) are probabilistic and "
                    "can disagree with each other or with themselves on repeated runs. Treat "
                    "any AI score as a rough signal, not evidence of who wrote a text."
        }

    except Exception as e:
        print("Humanize error:", e)
        return {"error": str(e)}

# ─────────────────────────────────────────
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
        c += f" {year}"
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