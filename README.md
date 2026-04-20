# Resume Enhancer — RAG
A Django web app that rewrites resume bullet points to be more impactful and ATS-friendly for a target job description. Built as a side-by-side experiment comparing four approaches: **Gemini with RAG**, **Gemini without RAG**, **Groq with RAG**, and **Groq without RAG** — each output scored against the JD using cosine similarity.

## Features

- **Four-way comparison.** Every submission produces 12 enhanced bullet points across two LLM providers × two pipelines × three styling angles (Keyword-Optimized, Achievement-Focused, Balanced).
- **RAG pipeline over a job-description corpus.** Incoming JDs are reformulated into a semantic query, matched against a local dataset with MiniLM embeddings, and re-ranked by a cross-encoder before generation.
- **Cosine similarity scoring.** Each generated bullet is embedded and scored against the target JD, with a color-coded similarity bar on the output page so you can eyeball which approach actually performs better.
- **Provider-level fault isolation.** A failure from one provider (rate limit, network hiccup) doesn't block the other — the page still renders with whatever completed successfully.


The RAG context is built once per request and reused across both providers, which keeps the comparison apples-to-apples.

## Tech stack

- **Backend:** Django 4.2
- **LLMs:** Google Gemini 2.5 Flash, Groq (Llama 3.3 70B)
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers`
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Similarity:** scikit-learn cosine similarity over dense embeddings
- **Frontend:** Bootstrap 5, vanilla templates

## Setup

### 1. Clone and set up the environment

```bash
git clone https://github.com/your-username/resume-enhancer-rag.git
cd resume-enhancer-rag

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

| Variable | Where to get it |
|---|---|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY` | [Groq Console](https://console.groq.com/keys) |
| `DJANGO_SECRET_KEY` | Run `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"` |

### 3. Add the job dataset

The RAG pipeline needs a `job_dataset.csv` file in the project root with the following columns:

```
Title, Skills, Responsibilities, Keywords, ExperienceLevel, YearsOfExperience
```

### 4. Run

```bash
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in your browser.


## Authors

- **B Rithvik**
- **B Havish**
