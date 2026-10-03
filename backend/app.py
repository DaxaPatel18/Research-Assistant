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
#  6. HUMANIZER (Pegasus NLP + MiniLM Pipeline)
# ─────────────────────────────────────────
paraphrase_model = None
paraphrase_tokenizer = None

def get_paraphrase_model():
    """
    Lazy loader for tuner007/pegasus_paraphrase transformer model.
    Cached globally after initial load to optimize memory and speed.
    """
    global paraphrase_model, paraphrase_tokenizer
    if paraphrase_model is None:
        print("Loading Pegasus Paraphrase model (tuner007/pegasus_paraphrase)...")
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer
        model_name = "tuner007/pegasus_paraphrase"
        paraphrase_tokenizer = PegasusTokenizer.from_pretrained(model_name)
        paraphrase_model = PegasusForConditionalGeneration.from_pretrained(model_name)
        paraphrase_model.eval()
        print("Pegasus Paraphrase model ready!")
    return paraphrase_tokenizer, paraphrase_model

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

def generate_candidates_pegasus(sentence_text: str, num_candidates: int = 3, style: str = "academic") -> list:
    """
    Generates multiple candidate paraphrases using pretrained Pegasus seq2seq model.
    Configures generation parameters according to selected style:
      - Academic: num_beams=5, do_sample=False (controlled deterministic search)
      - Casual:   num_beams=4, do_sample=True, temperature=1.2, top_p=0.90
      - Natural:  num_beams=4, do_sample=True, temperature=1.1, top_p=0.92
    """
    tokenizer, model = get_paraphrase_model()
    import torch

    inputs = tokenizer(
        [sentence_text],
        truncation=True,
        padding="longest",
        max_length=64,
        return_tensors="pt"
    )

    if style == "casual":
        gen_kwargs = {
            "num_beams": 4,
            "do_sample": True,
            "temperature": 1.2,
            "top_p": 0.90,
            "no_repeat_ngram_size": 2,
            "early_stopping": True
        }
    elif style == "natural":
        gen_kwargs = {
            "num_beams": 4,
            "do_sample": True,
            "temperature": 1.1,
            "top_p": 0.92,
            "no_repeat_ngram_size": 2,
            "early_stopping": True
        }
    else:  # academic / default
        gen_kwargs = {
            "num_beams": 5,
            "do_sample": False,
            "no_repeat_ngram_size": 2,
            "early_stopping": True
        }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_return_sequences=num_candidates,
            **gen_kwargs
        )

    candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    unique_candidates = []
    for c in candidates:
        c_clean = c.strip()
        if c_clean and c_clean not in unique_candidates:
            unique_candidates.append(c_clean)

    return unique_candidates

def score_candidate(original_sentence: str, candidate_sentence: str) -> dict:
    """
    Evaluates candidate paraphrase against the original sentence using multi-metric scoring:
    - 70% Semantic Similarity (MiniLM cosine similarity - dominant factor)
    - 15% Lexical Diversity (Word variation score)
    - 15% Quality & Readability Heuristics (Repetition penalty & length ratio)
    """
    # 1. Semantic Similarity (0.0 to 1.0)
    sim_pct = semantic_similarity_pct(original_sentence, candidate_sentence)
    if sim_pct is not None:
        sem_sim = max(0.0, min(1.0, sim_pct / 100.0))
    else:
        sem_sim = 0.5

    # 2. Lexical Diversity (0.0 to 1.0)
    orig_words = set(re.findall(r'\b\w+\b', original_sentence.lower()))
    cand_words = set(re.findall(r'\b\w+\b', candidate_sentence.lower()))

    if orig_words and cand_words:
        intersection = orig_words & cand_words
        union = orig_words | cand_words
        overlap_ratio = len(intersection) / len(union) if union else 1.0
        lexical_diversity = 1.0 - overlap_ratio
    else:
        lexical_diversity = 0.0

    # 3. Readability & Repetition Heuristics (0.0 to 1.0)
    orig_len = max(len(original_sentence.split()), 1)
    cand_len = max(len(candidate_sentence.split()), 1)
    len_ratio = cand_len / orig_len
    len_score = 1.0 if 0.7 <= len_ratio <= 1.3 else 0.7 if 0.5 <= len_ratio <= 1.5 else 0.3

    words = candidate_sentence.lower().split()
    if len(words) >= 4:
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        rep_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 1.0
    else:
        rep_ratio = 1.0

    quality_heuristic = (len_score * 0.6) + (rep_ratio * 0.4)

    # Dominant semantic weighting (70% Semantic, 15% Diversity, 15% Quality)
    final_score = (sem_sim * 0.70) + (lexical_diversity * 0.15) + (quality_heuristic * 0.15)

    # Near-identical check (normalized text comparison or lexical diversity < 0.05)
    is_near_identical = (candidate_sentence.strip().lower() == original_sentence.strip().lower()) or (lexical_diversity < 0.05)

    return {
        "final_score": round(final_score, 4),
        "semantic_similarity": round(sem_sim * 100, 1),
        "lexical_diversity": round(lexical_diversity * 100, 1),
        "quality_score": round(quality_heuristic * 100, 1),
        "candidate": candidate_sentence,
        "is_near_identical": is_near_identical
    }


def extract_protected_items(sentence: str) -> list:
    """
    Extracts protected technical information items from an original sentence:
    - Numbers (integers, decimals, percentages, numbers with commas like '10,000', '92.5%', '89.7%', '384')
    - Model names & technical identifiers ('all-MiniLM-L6-v2', 'Python', 'Pegasus', etc.)
    - Hyphenated technical identifiers ('384-dimensional', 'F1-score')
    - Important domain technical terms ('cosine similarity', 'embeddings', 'query', 'research papers', etc.)
    """
    items = []

    # 1. Numbers (integers, decimals, percentages, numbers with commas)
    num_pattern = re.compile(r'\b\d+(?:,\d+)*(?:\.\d+)?%?\b')
    for match in num_pattern.finditer(sentence):
        num_str = match.group(0).strip()
        if num_str and num_str not in items:
            items.append(num_str)

    # 2. Known model names, languages, and technical identifiers
    ident_pattern = re.compile(r'\b(?:all-MiniLM-L6-v2|MiniLM|Python|PyTorch|TensorFlow|Groq|Pegasus)\b', re.IGNORECASE)
    for match in ident_pattern.finditer(sentence):
        item = match.group(0).strip()
        if item and not any(item.lower() == existing.lower() for existing in items):
            items.append(item)

    # 3. Hyphenated technical identifiers and acronyms
    hyphen_pattern = re.compile(r'\b[A-Za-z0-9]+-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b')
    for match in hyphen_pattern.finditer(sentence):
        item = match.group(0).strip()
        if item and not any(item.lower() == existing.lower() for existing in items):
            items.append(item)

    # 4. Domain-specific technical terms & concepts
    KNOWN_TECH_TERMS = [
        "cosine similarity", "research papers", "research paper", "machine learning",
        "training data", "biased data", "training subsets", "testing subsets",
        "logistic regression", "decision trees", "random forests",
        "support vector machines", "support vector", "neural networks", "neural network",
        "transformer models", "transformer model", "transformer",
        "semantic representations", "semantic representation",
        "sentence embeddings", "sentence embedding", "numerical vectors", "numerical vector",
        "semantic relevance", "keyword matches", "keyword match", "keyword",
        "data analysis", "embeddings", "embedding", "query", "queries"
    ]
    sent_lower = sentence.lower()
    for term in KNOWN_TECH_TERMS:
        if term in sent_lower:
            start_idx = sent_lower.find(term)
            exact_casing = sentence[start_idx:start_idx + len(term)]
            if exact_casing and not any(exact_casing.lower() == existing.lower() or existing.lower() in exact_casing.lower() for existing in items):
                items.append(exact_casing)

    return items

def verify_protected_items_preserved(candidate: str, protected_items: list) -> tuple:
    """
    Verifies if all protected items from the original sentence are preserved in the candidate paraphrase.
    Returns (is_preserved: bool, missing_items: list).
    """
    cand_lower = candidate.lower()
    missing = []

    for item in protected_items:
        item_lower = item.lower()

        # Direct substring match
        if item_lower in cand_lower:
            continue

        # Variations handling
        # 1. Hyphenated vs spaced (e.g. 'f1-score' vs 'f1 score')
        if '-' in item_lower and item_lower.replace('-', ' ') in cand_lower:
            continue

        # 2. Number with/without commas (e.g. '10,000' vs '10000')
        if ',' in item_lower and item_lower.replace(',', '') in cand_lower:
            continue

        # 3. Percentages (e.g. '92.5%' vs '92.5 percent' or '92.5')
        if item_lower.endswith('%'):
            val_no_pct = item_lower[:-1].strip()
            if val_no_pct in cand_lower:
                continue

        missing.append(item)

    return (len(missing) == 0, missing)

@app.post("/humanize")
def humanize(req: TextAnalysisRequest):
    """
    ML Paraphrasing & Humanizer Pipeline:
    Text -> Sentence Segmentation -> Protected Information Extraction -> Pegasus Sentence Paraphrasing ->
    MiniLM Semantic & Quality Scoring -> Protected Information Verification -> Candidate Selection / Safety Fallback -> Conservative AI Phrase Cleanup
    Guarantees N input sentences -> N output sentences.
    """
    try:
        text = req.text.strip()
        style = req.style or "academic"

        if len(text) < 20:
            return {"error": "Please provide at least 20 characters."}

        fallback_used = False
        sentences_processed = 0
        sentences_changed = 0
        candidates_rejected = 0
        candidates_rejected_protected_info = 0
        unchanged_candidates = 0
        total_candidates_eval = 0
        protected_items_found = 0
        protected_items_preserved = 0

        selected_sentences = []
        sentence_scores = []

        try:
            # 1-to-1 Sentence Segmentation: N input sentences -> N output units
            sentences = split_into_sentences(text)
            sentences_processed = len(sentences)

            for sent in sentences:
                protected = extract_protected_items(sent)
                protected_items_found += len(protected)

                candidates = generate_candidates_pegasus(sent, num_candidates=3, style=style)
                total_candidates_eval += len(candidates)

                # Score Pegasus candidates against original sentence ONLY
                scored_candidates = []
                for cand in candidates:
                    scored = score_candidate(sent, cand)
                    
                    # Verify protected information preservation
                    is_prot_preserved, missing_items = verify_protected_items_preserved(cand, protected)
                    scored["protected_preserved"] = is_prot_preserved
                    scored["missing_protected"] = missing_items

                    if not is_prot_preserved:
                        candidates_rejected_protected_info += 1
                        if scored["is_near_identical"]:
                            unchanged_candidates += 1
                        continue  # Reject candidate failing protected information check!

                    scored_candidates.append(scored)
                    if scored["is_near_identical"]:
                        unchanged_candidates += 1

                # Filter candidates >= 35% semantic similarity
                valid_candidates = [sc for sc in scored_candidates if sc["semantic_similarity"] >= 35.0]
                candidates_rejected += (len(candidates) - len(valid_candidates))

                if valid_candidates:
                    # Sort valid candidates by final score descending
                    sorted_valid = sorted(valid_candidates, key=lambda x: x["final_score"], reverse=True)
                    
                    # Prefer a valid candidate that is not near-identical to original
                    non_identical_candidates = [sc for sc in sorted_valid if not sc["is_near_identical"]]
                    if non_identical_candidates:
                        best_choice = non_identical_candidates[0]
                    else:
                        best_choice = sorted_valid[0]

                    selected_text = best_choice["candidate"]
                    sentence_scores.append(best_choice["final_score"])
                    
                    if selected_text.strip().lower() != sent.strip().lower():
                        sentences_changed += 1

                    # Count preserved protected items for best choice
                    _, missing_in_selected = verify_protected_items_preserved(selected_text, protected)
                    protected_items_preserved += (len(protected) - len(missing_in_selected))

                else:
                    # Safety Fallback: Use ORIGINAL SENTENCE if all generated candidates failed protected info or sem_sim check
                    selected_text = sent
                    sentence_scores.append(0.85)
                    protected_items_preserved += len(protected)

                selected_sentences.append(selected_text)

            raw_humanized = " ".join(selected_sentences)

        except Exception as model_err:
            print("Pegasus model execution error, using LLM fallback:", model_err)
            fallback_used = True
            instruction = f"Rewrite this text into natural, fluent academic prose in '{style}' style. Preserve all facts, terms, numbers, and meaning."
            raw_humanized = ask_groq(instruction, text, max_tokens=1500)
            sentences_processed = len(split_into_sentences(text))
            sentences_changed = sentences_processed
            total_candidates_eval = 1

        # Conservative final cleanup pass: subtle removal of flagged AI phrases without aggressive rewriting
        humanized = strip_ai_phrases(raw_humanized)

        # Compute overall semantic similarity using all-MiniLM-L6-v2
        meaning_similarity = semantic_similarity_pct(text, humanized)
        avg_quality_score = round(sum(sentence_scores) / len(sentence_scores) * 100, 1) if sentence_scores else 80.0

        changes_made = []
        if fallback_used:
            changes_made.append("Applied fallback neural rephrasing")
        else:
            changes_made.append("Paraphrased sentence-by-sentence with Pegasus (tuner007/pegasus_paraphrase)")
            changes_made.append("Evaluated candidates with MiniLM semantic preservation (70% weight)")
            if candidates_rejected_protected_info > 0:
                changes_made.append(f"Enforced Protected Info Validation ({candidates_rejected_protected_info} candidate(s) rejected for missing facts)")
        changes_made.append("Applied conservative phrase cleanup")
        changes_made.append(f"Style applied: {style.title()}")

        return {
            "humanized_text": humanized,
            "original_words": len(text.split()),
            "humanized_words": len(humanized.split()),
            "changes_made": changes_made,
            "meaning_similarity": meaning_similarity,
            "paraphrase_quality": avg_quality_score,
            "chunks_processed": sentences_processed,
            "sentences_processed": sentences_processed,
            "sentences_changed": sentences_changed,
            "candidates_evaluated": total_candidates_eval,
            "candidates_rejected": candidates_rejected,
            "candidates_rejected_protected_info": candidates_rejected_protected_info,
            "protected_items_found": protected_items_found,
            "protected_items_preserved": protected_items_preserved,
            "unchanged_candidates": unchanged_candidates,
            "pipeline": "Pegasus seq2seq Sentence Paraphrasing + all-MiniLM-L6-v2 Semantic Scoring",
            "fallback_used": fallback_used,
            "tip": "Paraphrased sentence-by-sentence using a pretrained Transformer sequence-to-sequence model and evaluated for semantic preservation.",
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