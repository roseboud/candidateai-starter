# CandidateAI starter

A small retrieval-augmented generation (RAG) app. It answers recruiters' questions about one person, using their resume and a short profile as its only source. It's the technical lane of [Building RAG from a Resume](https://quinan.tech/rag/), a free course from Quinan Labs, first taught at Saint Mary's University.

The OpenRouter key lives in an environment variable on the server, never in the page. For paid models, every request asks OpenRouter to use only providers that keep nothing and don't train on what they receive.

## Files

| File | What it does |
| --- | --- |
| `app.py` | The server. Receives a question, runs the search step, builds the prompt, calls OpenRouter. |
| `retrieval.py` | The search step. Splits the documents into chunks and scores them against the question. |
| `static/index.html` | The page visitors see. |
| `data/resume.txt` | Your resume as plain text. The sample is a made-up student. Replace it. |
| `data/profile.txt` | A short profile in your own words. Replace it too. |
| `requirements.txt` | The Python packages to install. |
| `.python-version` | Tells Render to use Python 3.12. |
| `.env.example` | A template for the secrets file you use on your laptop. |

## Deploy on Render

1. Put your text in `data/resume.txt` and `data/profile.txt` and commit.
2. In Render, choose **New > Web Service** and connect this repository.
3. Set **Language** to Python 3, **Build Command** to `pip install -r requirements.txt`, **Start Command** to `uvicorn app:app --host 0.0.0.0 --port $PORT`, and **Instance Type** to Free.
4. Under **Environment Variables**, add `OPENROUTER_KEY` (your key) and `CANDIDATE_NAME` (your name).
5. Deploy. Your link ends in `onrender.com`. Free services sleep after 15 minutes without visitors and take about a minute to wake.

## Run on your laptop (optional)

Needs Python 3.10 or newer. Open a terminal in the folder that holds `app.py`.

Windows (PowerShell):

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

macOS or Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and paste your key. Then start the app, and open http://127.0.0.1:8000 when it says it's running.

Windows:

```
.venv\Scripts\python -m uvicorn app:app --reload
```

macOS or Linux:

```
uvicorn app:app --reload
```

## Settings

These are environment variables. Set them in `.env` on your laptop or in Render's Environment tab.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENROUTER_KEY` | none | Required. Your OpenRouter API key. Never put it in the code. |
| `CANDIDATE_NAME` | Jordan Doucette | Your name, as the page and the model should use it. |
| `MODEL` | google/gemini-3.1-flash-lite | Any model ID from openrouter.ai/models. |
| `USE_RETRIEVAL` | true | Set to `false` to send every chunk with every question, and compare the answers. |

## Things to try

- Set `USE_RETRIEVAL=false` and ask the same questions. For one resume, the answers barely change. Why would that stop being true for 5,000 documents?
- Ask "Where did Jordan study?" and open the sources. Keyword search matched the word "Study" in a project title. Search by meaning (embeddings) would find the education section.
- Send the last few questions and answers along with each new question, so follow-ups like "tell me more about that" work.
- Change `INSTRUCTIONS` in `app.py` and see which rules the model keeps and which it bends.
