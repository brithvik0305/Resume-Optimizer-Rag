import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render, redirect
from django.conf import settings

GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
)

GROQ_API_KEY = settings.GROQ_API_KEY
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  

def call_gemini(prompt: str) -> str:
    response = requests.post(
        GEMINI_URL,
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Gemini: {data['error']['message']}")
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def call_groq(prompt: str) -> str:
    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        timeout=60,
    )
    data = response.json()
    if "error" in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError(f"Groq: {msg}")
    return data["choices"][0]["message"]["content"].strip()


def parse_versions(raw: str) -> dict:
    """Strip markdown fences and parse JSON from LLM output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "version_1": raw,
            "version_2": "",
            "version_3": "",
        }


def _load_rag_components():
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(BASE_DIR, "job_dataset.csv")

    df = pd.read_csv(dataset_path)
    df["combined_text"] = (
        "Title: "              + df["Title"].fillna("")             + ". "
        + "Skills: "           + df["Skills"].fillna("")            + ". "
        + "Responsibilities: " + df["Responsibilities"].fillna("")   + ". "
        + "Keywords: "         + df["Keywords"].fillna("")
    )

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    job_embeddings  = embedding_model.encode(
        df["combined_text"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return df, embedding_model, job_embeddings, reranker


try:
    _df, _embedding_model, _job_embeddings, _reranker = _load_rag_components()
    RAG_READY = True
    print("[RAG] Components loaded successfully")
except Exception as _e:
    RAG_READY = False
    import traceback
    print(f"[RAG] Failed to load components: {_e}")
    traceback.print_exc()


def reformulate_query(job_description: str) -> str:
    prompt = (
        "You are a technical recruiter. Distill the following job description "
        "into a concise search query capturing the core skills, technologies, "
        "and role requirements.\n"
        "Return ONLY the query as a single line of comma-separated terms. No explanation.\n\n"
        f"Job Description:\n{job_description[:1000]}"
    )
    return call_gemini(prompt)


def retrieve_relevant_jobs(query_text: str, top_k: int = 10) -> list:
    query_embedding = _embedding_model.encode([query_text], convert_to_numpy=True)
    similarities    = cosine_similarity(query_embedding, _job_embeddings).flatten()
    top_indices     = similarities.argsort()[-top_k:][::-1]

    return [
        {
            "title":            _df.iloc[idx]["Title"],
            "experience_level": _df.iloc[idx]["ExperienceLevel"],
            "years":            _df.iloc[idx]["YearsOfExperience"],
            "skills":           _df.iloc[idx]["Skills"],
            "responsibilities": _df.iloc[idx]["Responsibilities"],
            "keywords":         _df.iloc[idx]["Keywords"],
            "similarity_score": float(similarities[idx]),
        }
        for idx in top_indices
    ]


def rerank_jobs(query: str, candidates: list, top_n: int = 3) -> list:
    pairs  = [
        (query, f"{j['title']}. Skills: {j['skills']}. Keywords: {j['keywords']}")
        for j in candidates
    ]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    results = []
    for score, job in ranked[:top_n]:
        job = dict(job)
        job["rerank_score"] = float(score)
        results.append(job)
    return results


JSON_INSTRUCTION = """
Return ONLY a valid JSON object with exactly these keys, no markdown, no explanation:
{
  "version_1": "<Keyword-Optimized bullet point>",
  "version_2": "<Achievement-Focused bullet point>",
  "version_3": "<Balanced bullet point>"
}
"""

def build_rag_prompt(bullet_point: str, job_description: str,
                     retrieve_k: int = 10, final_k: int = 3) -> tuple:
    expanded_query = reformulate_query(job_description)
    candidates     = retrieve_relevant_jobs(expanded_query, top_k=retrieve_k)
    relevant_jobs  = rerank_jobs(expanded_query, candidates, top_n=final_k)

    context = "RELEVANT JOB DESCRIPTIONS FROM DATASET:\n\n"
    for i, job in enumerate(relevant_jobs, 1):
        context += f"{i}. {job['title']} ({job['experience_level']})\n"
        context += f"   Key Skills: {job['skills']}\n"
        context += f"   Keywords: {job['keywords']}\n"
        context += f"   Responsibilities: {job['responsibilities'][:200]}...\n\n"

    prompt = f"""You are an expert resume writer. Enhance the following resume bullet point \
to make it more impactful and ATS-friendly for the given job description.

ORIGINAL BULLET POINT:
{bullet_point}

TARGET JOB DESCRIPTION:
{job_description[:500]}

{context}

TASK:
1. Enhance the bullet point using keywords from both the target JD and retrieved job descriptions
2. Use strong action verbs and quantifiable achievements where possible
3. Ensure ATS-friendly, industry-standard terminology tailored to the target JD
4. Keep the core accomplishment but make it more compelling
5. No bullet symbols, no markdown, plain text only

{JSON_INSTRUCTION}"""

    return prompt, expanded_query, relevant_jobs


def build_normal_prompt(bullet_point: str, job_description: str) -> str:
    return f"""You are an expert resume writer. Enhance the following resume bullet point \
to make it more impactful and ATS-friendly for the given job description.

ORIGINAL BULLET POINT:
{bullet_point}

TARGET JOB DESCRIPTION:
{job_description[:500]}

TASK:
1. Use strong action verbs and quantifiable achievements where possible
2. ATS-friendly with industry-standard terminology
3. Keep the core accomplishment but make it more compelling
4. No bullet symbols, no markdown, plain text only

{JSON_INSTRUCTION}"""


EMPTY_VERSIONS = {"version_1": "", "version_2": "", "version_3": ""}
RAG_UNAVAILABLE = {
    "version_1": "RAG components not available.",
    "version_2": "",
    "version_3": "",
}


def _safe_call(caller, prompt, label):
    """Wrap a provider call so one failure doesn't kill the other provider."""
    try:
        return parse_versions(caller(prompt))
    except Exception as exc:
        return {
            "version_1": f"[{label} error] {exc}",
            "version_2": "",
            "version_3": "",
        }



def _empty_scored(text: str) -> dict:
    """Build the scored-dict shape used by the template when scoring is skipped."""
    return {
        "text":          text or "",
        "score":         None,
        "score_pct":     0,
        "score_display": "—",
    }


def score_versions(versions: dict, jd_embedding) -> dict:
    """
    Attach cosine similarity (version vs. JD) to each version.

    Input:  {"version_1": "text", "version_2": "text", "version_3": "text"}
    Output: {"version_1": {"text": ..., "score": 0.84, "score_pct": 84, "score_display": "0.842"}, ...}
    """
    keys  = ["version_1", "version_2", "version_3"]
    texts = [(versions.get(k) or "").strip() for k in keys]

    if jd_embedding is None or not RAG_READY:
        return {k: _empty_scored(versions.get(k, "")) for k in keys}

    non_empty_idx = [i for i, t in enumerate(texts) if t]
    scores = [None] * len(keys)

    if non_empty_idx:
        embs = _embedding_model.encode(
            [texts[i] for i in non_empty_idx],
            convert_to_numpy=True,
        )
        sims = cosine_similarity(jd_embedding, embs).flatten()
        for pos, idx in enumerate(non_empty_idx):
            scores[idx] = float(sims[pos])

    out = {}
    for i, key in enumerate(keys):
        score = scores[i]
        if score is None:
            out[key] = _empty_scored(versions.get(key, ""))
        else:
            pct = max(0, min(100, int(round(score * 100))))
            out[key] = {
                "text":          versions.get(key, ""),
                "score":         score,
                "score_pct":     pct,
                "score_display": f"{score:.3f}",
            }
    return out



def home(request):
    if request.method == "POST":
        bullet_points   = request.POST.get("bullet_points", "").strip()
        job_description = request.POST.get("job_description", "").strip()

        if not bullet_points or not job_description:
            return render(request, "home.html", {
                "error": "Please fill in both fields."
            })

        rag_prompt = None
        if RAG_READY:
            try:
                rag_prompt, _, _ = build_rag_prompt(bullet_points, job_description)
            except Exception as e:
                return render(request, "home.html", {
                    "error": f"RAG pipeline failed: {e}"
                })

        normal_prompt = build_normal_prompt(bullet_points, job_description)

        gemini_rag    = _safe_call(call_gemini, rag_prompt, "Gemini") if rag_prompt else RAG_UNAVAILABLE
        gemini_normal = _safe_call(call_gemini, normal_prompt, "Gemini")

        groq_rag      = _safe_call(call_groq, rag_prompt, "Groq") if rag_prompt else RAG_UNAVAILABLE
        groq_normal   = _safe_call(call_groq, normal_prompt, "Groq")

        jd_emb = None
        if RAG_READY:
            try:
                jd_emb = _embedding_model.encode(
                    [job_description], convert_to_numpy=True
                )
            except Exception as e:
                print(f"[similarity] JD embedding failed: {e}")

        gemini_rag    = score_versions(gemini_rag,    jd_emb)
        gemini_normal = score_versions(gemini_normal, jd_emb)
        groq_rag      = score_versions(groq_rag,      jd_emb)
        groq_normal   = score_versions(groq_normal,   jd_emb)

        request.session["gemini_rag"]    = gemini_rag
        request.session["gemini_normal"] = gemini_normal
        request.session["groq_rag"]      = groq_rag
        request.session["groq_normal"]   = groq_normal
        request.session["groq_model"]    = GROQ_MODEL
        return redirect("output")

    return render(request, "home.html")


EMPTY_SCORED_SET = {
    "version_1": _empty_scored(""),
    "version_2": _empty_scored(""),
    "version_3": _empty_scored(""),
}


def output(request):
    ctx = {
        "gemini_rag":    request.session.pop("gemini_rag",    EMPTY_SCORED_SET),
        "gemini_normal": request.session.pop("gemini_normal", EMPTY_SCORED_SET),
        "groq_rag":      request.session.pop("groq_rag",      EMPTY_SCORED_SET),
        "groq_normal":   request.session.pop("groq_normal",   EMPTY_SCORED_SET),
        "groq_model":    request.session.pop("groq_model",    GROQ_MODEL),
    }
    return render(request, "output.html", ctx)
