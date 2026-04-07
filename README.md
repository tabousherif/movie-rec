# Reelio 🎬

**AI-powered semantic movie recommendation engine.**  
Search by vibe, director, actor, or title — Reelio finds what to watch next using embeddings, vector search, and LLM reranking.

🔗 [Live Demo](#) &nbsp;·&nbsp; [Backend API Docs](http://localhost:8000/docs)

---

## What It Does

You pick 1–2 seed movies. Reelio:
1. Looks up their semantic embedding vectors
2. Averages them into a single query vector
3. Searches 5,000+ movies via FAISS to find the 25 closest matches
4. Passes those candidates to Claude, which reranks to 5 personalized recommendations with explanations
5. Displays results with rec scores, TMDB ratings, and Canadian streaming availability (Netflix, Prime Video, Disney+, Crave, and more)

---

## Architecture

```
User Input (seed movies + preferences)
        │
        ▼
┌─────────────────────┐
│   sentence-         │  Embeds seed movie descriptions
│   transformers      │  into 384-dim vectors
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   FAISS Index       │  Nearest-neighbor search across
│   (IndexFlatIP)     │  5,000+ movie embeddings
└────────┬────────────┘
         │  top 25 candidates
         ▼
┌─────────────────────┐
│   Claude API        │  Reranks candidates based on
│   (Reranker)        │  user preferences + assigns
│                     │  rec scores + writes explanations
└────────┬────────────┘
         │  top 5 results
         ▼
┌─────────────────────┐
│   FastAPI Backend   │  Serves /search and /recommend
│   + Pydantic        │  with full metadata + streaming
└────────┬────────────┘
         │
         ▼
    Reelio Frontend
    (Vanilla JS SPA)
```

### Why This Approach Over Content-Based Filtering

The alternative was a classical content-based filtering approach — manually engineer feature vectors (genre, runtime, rating, decade) and compute cosine similarity. That was rejected for two reasons:

- **Semantic gap:** Feature vectors can identify that *Inception* and *Interstellar* share a director and genre, but can't capture that *Parasite* and *Knives Out* share a thematic DNA around class tension and dark comedy despite completely different genres, eras, and directors. Sentence-transformer embeddings capture meaning from natural language descriptions, not just structured attributes.
- **No reranking intelligence:** A feature vector model has no way to factor in "I only have 90 minutes" or "watching with my parents." The LLM reranking layer handles this naturally.

The tradeoff is external API dependency and slightly higher latency. For a production system, the embedding + FAISS layer would handle real-time traffic while Claude reranking could be made async or cached.

---

## ML Pipeline

The data pipeline runs once offline and produces three artifacts loaded into memory at server startup:

| Script | Output | Purpose |
|---|---|---|
| `fetch_movies.py` | `data/movies.json` | 5,000+ movies from TMDB API with metadata |
| `embed_movies.py` | `data/embeddings.npy` | (5000, 384) float32 embedding matrix |
| `build_index.py` | `data/movies.index` | FAISS IndexFlatIP, query-ready |
| `enrich_streaming.py` | updates `movie_index.json` | Canadian streaming availability per movie |

**Embedding model:** `all-MiniLM-L6-v2` — 384 dimensions, ~90MB, strong semantic quality at fast inference speed. Embeddings are L2-normalized so inner product == cosine similarity.

**Query construction:** Seed movie vectors are averaged into a single query vector, then re-normalized. This finds movies that share characteristics with *both* seeds rather than just one.

**Scoring:** Raw FAISS cosine similarity scores (0–1) are normalized to a 55–100 range relative to each batch. Claude then assigns a final 1–100 rec score based on preference fit, which is what appears on the card.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Search | `FAISS` (IndexFlatIP) |
| LLM Reranking | Anthropic Claude API (claude-sonnet) |
| Backend | FastAPI + Pydantic + Uvicorn |
| Data | TMDB API (movies + streaming providers) |
| Frontend | Vanilla JS SPA (no framework) |
| Hosting | Render (backend) · Vercel (frontend) |

---

## Project Structure

```
movie-rec/
├── backend/
│   ├── data/
│   │   ├── movies.json          # raw TMDB metadata
│   │   ├── movie_index.json     # enriched lookup table (embeddings position → movie)
│   │   ├── embeddings.npy       # float32 embedding matrix
│   │   ├── movies.index         # FAISS index
│   │   └── streaming_cache.json # CA streaming enrichment cache
│   ├── fetch_movies.py          # step 1: pull TMDB data
│   ├── embed_movies.py          # step 2: generate embeddings
│   ├── build_index.py           # step 3: build FAISS index
│   ├── enrich_streaming.py      # step 4: add CA streaming availability
│   ├── recommender.py           # FAISS retrieval + scoring
│   ├── claude_reranker.py       # Claude reranking + explanation
│   ├── main.py                  # FastAPI app
│   └── requirements.txt
└── frontend/
    └── index.html               # single-page frontend
```

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ (if converting frontend to React)
- TMDB API key → [themoviedb.org](https://www.themoviedb.org/settings/api)
- Anthropic API key → [console.anthropic.com](https://console.anthropic.com)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Edit .env with your TMDB_API_KEY and ANTHROPIC_API_KEY
```

### Build the ML pipeline (one-time)

```bash
python fetch_movies.py          # ~5-8 min  — pulls 5,000 movies from TMDB
python embed_movies.py          # ~3-4 min  — generates embeddings
python build_index.py           # ~5 sec    — builds FAISS index
python enrich_streaming.py      # ~25-35 min — adds CA streaming data (resumable)
```

### Run the server

```bash
uvicorn main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### Frontend

Open `frontend/index.html` directly in a browser, or serve it:

```bash
npx serve frontend/
```

> **Note:** Update `API_BASE` at the top of `index.html` to point to your deployed backend URL before deploying the frontend.

---

## API Reference

### `GET /search?q={query}&limit={n}`
Autocomplete search against movie titles. Returns lightweight results for the search bar dropdown.

### `POST /recommend`
Full recommendation pipeline.

```json
{
  "seeds": [
    { "id": 27205, "title": "Inception" },
    { "id": 603,   "title": "The Matrix" }
  ],
  "preferences": {
    "genre":       "Sci-Fi",
    "runtime_max": 120,
    "group":       "date night",
    "mood":        "mind-bending"
  },
  "use_mock": false
}
```

Set `use_mock: true` to skip the Claude API call during development (uses FAISS top-5 order with placeholder explanations, costs $0).

---

## Deployment

**Backend → [Render](https://render.com)**
- Connect your GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add `TMDB_API_KEY` and `ANTHROPIC_API_KEY` as environment variables
- Note: `embeddings.npy` and `movies.index` are large — add them to your repo or use Render's persistent disk

**Frontend → [Vercel](https://vercel.com)**
- Connect your GitHub repo, set root to `frontend/`
- Update `API_BASE` in `index.html` to your Render service URL before deploying

---

## Known Limitations

- **Streaming data freshness:** TMDB watch provider data has a lag — content rotates on/off platforms and may not reflect same-day changes
- **Dataset size:** 5,000 movies skews toward popular/well-rated titles. Niche or older films may not be in the index
- **Latency:** First request after server cold-start is slower as ML artifacts load into memory (~3–5s). Subsequent requests are fast
- **Region:** Streaming availability is Canadian only (`CA` region via TMDB)

---

## License

MIT
