# Resume Enhancer — RAG
A Django web app that rewrites resume bullet points to be more impactful and ATS-friendly for a target job description. Built as a side-by-side experiment comparing four approaches: **Gemini with RAG**, **Gemini without RAG**, **Groq with RAG**, and **Groq without RAG** — each output scored for JD keyword alignment and checked for skills the model invented.

## Features

- **Four-way comparison.** Every submission produces 12 enhanced bullet points across two LLM providers × two pipelines × three styling angles (Keyword-Optimized, Achievement-Focused, Balanced). Generation configs are matched across providers — same temperature (0.7), same max output tokens, Gemini's thinking mode disabled — so differences reflect the provider and pipeline, not decoding settings.
- **RAG pipeline over a job-description corpus.** Incoming JDs are reformulated into a semantic query, matched against a local dataset with MiniLM embeddings, and re-ranked by a cross-encoder before generation. If query reformulation fails (it's an LLM call), retrieval falls back to the raw JD instead of failing the request.
- **JD Keyword Alignment score.** Each generated bullet is embedded and scored with cosine similarity against the target JD, with a color-coded bar on the output page. This is deliberately labeled *alignment*, not *quality*: the RAG prompt injects JD-adjacent vocabulary into generation, so the metric is structurally biased toward RAG outputs and keyword-heavy phrasing. Read it as "how much JD vocabulary landed in this bullet" and weigh it against the faithfulness badge below.
- **Faithfulness badge (hallucination check).** Every bullet is checked against a skill vocabulary derived from the corpus's Skills/Keywords columns; any skill or tool that appears in the output but not in your original bullet is flagged inline ("⚠ adds: …"). Alignment tells you a bullet matches the JD; the badge tells you whether it stayed true to what you actually did.
- **Structured output via native JSON mode.** Both providers are called with JSON output enforced (Gemini `responseMimeType`, Groq `response_format`), and responses are validated for the expected keys. A cell that still fails renders as an explicit error state and is excluded from scoring — error text never receives a similarity score.
- **Provider-level fault isolation.** The four generation calls run concurrently in a thread pool and each failure is contained to its own cell — a Gemini rate limit doesn't block Groq's results, and vice versa. If the retrieval stack is unavailable, only the two RAG cells degrade.

Input handling: the job description is capped at 1,200 characters (truncated at a word boundary), and the **same slice** is used for query reformulation, both generation prompts, and similarity scoring — so the score is computed over text the models actually saw. The RAG context is built once per request and reused across both providers, which keeps the comparison apples-to-apples.

## Tech stack

- **Backend:** Django 4.2+
- **LLMs:** Google Gemini 2.5 Flash, Groq (Llama 3.3 70B)
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers`
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Similarity:** scikit-learn cosine similarity over dense embeddings
- **Frontend:** Bootstrap 5, vanilla templates

## Setup

### 1. Clone and set up the environment

```bash
git clone https://github.com/brithvik0305/Resume-Optimizer-Rag.git
cd Resume-Optimizer-Rag

python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note on PyTorch:** `torch` is a large download (~800 MB). On CPU-only machines, install the CPU wheel first to avoid accidentally pulling CUDA builds:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Required — get one at [Google AI Studio](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY` | Required — get one at [Groq Console](https://console.groq.com/keys) |
| `DJANGO_SECRET_KEY` | Optional for local development (an insecure hardcoded fallback is used); set it for anything beyond that. Generate with `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"` |
| `HF_OFFLINE` | Optional — set to `1` to force Hugging Face offline mode once the models are cached locally. Leave unset/`0` on first run so they can download. |

### 3. The job dataset

A `job_dataset.csv` with ~1,000 role profiles ships with the repo. It powers both retrieval **and** the faithfulness skill vocabulary. To swap in your own, keep the same columns (`Skills`/`Keywords` are semicolon-separated):

```
JobID, Title, ExperienceLevel, YearsOfExperience, Skills, Responsibilities, Keywords
```

### 4. Run

```bash
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in your browser.

> **First run:** the embedding model and cross-encoder (~200 MB combined) are downloaded from Hugging Face, and the corpus is embedded at startup — expect the first launch to take a minute or two. If the models can't be loaded, the app degrades gracefully: generation still works, but the RAG cells and similarity scores are disabled.


## Authors

- **B Rithvik**
- **B Havish**
