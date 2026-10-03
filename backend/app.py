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

        ai_phrases = [
            "in conclusion", "it is worth noting", "furthermore",
            "it is important to", "in summary", "to summarize",
            "in the realm of", "delve into", "it's worth noting",
            "as an ai", "certainly", "absolutely", "of course",
            "in today's world", "it is crucial", "plays a crucial role",
            "a testament to", "in the ever-evolving", "it is essential",
            "needless to say", "as previously mentioned", "it goes without saying"
        ]
        text_lower      = text.lower()
        ai_phrase_count = sum(1 for phrase in ai_phrases if phrase in text_lower)

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
    try:
        text  = req.text.strip()
        style = req.style or "academic"

        if len(text) < 50:
            return {"error": "Please provide at least 50 characters."}

        # Different prompts for different humanization styles
        style_instructions = {
            "academic": """You are rewriting AI-generated academic text to sound like
a genuine human student or researcher wrote it.
Rules:
1. Vary sentence lengths — alternate between short punchy sentences and longer ones
2. Add contractions naturally (don't, it's, we've, they're, isn't, wasn't)
3. Replace AI transitions: remove "furthermore","moreover","it is worth noting","in conclusion"
   Replace with: "also","and","but","so","though","still","even so"
4. Add hedging and personal perspective: "I think","it seems","arguably","perhaps","in my view"
5. Occasionally start sentences with "And" or "But" — humans do this, AI avoids it
6. Use simpler words: "utilize"→"use", "demonstrate"→"show", "facilitate"→"help"
7. Break up long uniform paragraphs
8. Keep ALL key information, facts, and meaning intact""",

            "casual": """You are rewriting AI text to sound like a casual but smart student wrote it.
Rules:
1. Use lots of contractions (don't, it's, they're, we've, can't, won't)
2. Add informal transitions: "basically","in short","the thing is","what's interesting is"
3. Use shorter sentences — aim for mix of 5-word and 20-word sentences
4. Add occasional filler that sounds natural: "actually","pretty much","kind of","in a way"
5. Remove ALL formal AI phrases like "it is imperative","it is worth noting","delve into"
6. Add personal voice: "I'd argue","honestly","to be fair","from what I can tell"
7. Keep all the information but make it feel conversational
8. Occasionally ask rhetorical questions""",

            "natural": """You are rewriting AI text to sound like a naturally thoughtful person wrote it.
Rules:
1. Create natural rhythm — some very short sentences. Some longer ones that flow naturally.
2. Add natural imperfections: occasional parenthetical thoughts (like this one)
3. Use em dashes for asides — they feel human
4. Contractions throughout: don't, it's, that's, we're, they've
5. Remove robotic precision: instead of "there are three key factors" say "a few things stand out"
6. Add genuine-sounding opinions: "what's surprising here","this is where it gets interesting"
7. Vary paragraph length — one sentence paragraphs are fine
8. Keep all facts and core meaning"""
        }

        instruction = style_instructions.get(style, style_instructions["academic"])

        humanized = ask_groq(
            instruction,
            f"""Rewrite the following text to sound naturally human-written.
Do NOT add any explanation or preamble — just output the rewritten text directly.

Original text:
{text}

Rewritten version:""",
            max_tokens=2000
        )

        # Detect what changed
        original_words   = set(text.lower().split())
        humanized_words  = set(humanized.lower().split())
        contractions     = ["don't","it's","we've","they're","isn't","wasn't","can't","won't","that's"]
        contractions_added = [c for c in contractions if c in humanized.lower() and c not in text.lower()]

        ai_phrases_removed = []
        ai_phrases = ["furthermore","moreover","it is worth noting","in conclusion",
                      "it is important to","in summary","delve into","it is crucial",
                      "needless to say","it goes without saying"]
        for phrase in ai_phrases:
            if phrase in text.lower() and phrase not in humanized.lower():
                ai_phrases_removed.append(phrase)

        changes_made = []
        if contractions_added:
            changes_made.append(f"Added contractions: {', '.join(contractions_added[:4])}")
        if ai_phrases_removed:
            changes_made.append(f"Removed AI phrases: {', '.join(ai_phrases_removed[:4])}")

        orig_sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        hum_sents  = [s for s in re.split(r'[.!?]+', humanized) if s.strip()]
        if len(orig_sents) > 1 and len(hum_sents) > 1:
            orig_lens = [len(s.split()) for s in orig_sents]
            hum_lens  = [len(s.split()) for s in hum_sents]
            orig_var  = sum((l - sum(orig_lens)/len(orig_lens))**2 for l in orig_lens) / len(orig_lens)
            hum_var   = sum((l - sum(hum_lens)/len(hum_lens))**2 for l in hum_lens) / len(hum_lens)
            if hum_var > orig_var:
                changes_made.append("Increased sentence length variety")

        if not changes_made:
            changes_made.append("Rewrote with more natural flow and human voice")
        changes_made.append(f"Style applied: {style.title()}")

        return {
            "humanized_text":  humanized,
            "original_words":  len(text.split()),
            "humanized_words": len(humanized.split()),
            "changes_made":    changes_made,
            "tip": "Click 'Test AI Score' to run detection on the humanized text and see your improved score."
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