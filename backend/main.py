"""
main.py
───────
Reelio FastAPI backend.

Endpoints:
  GET  /                          health check
  GET  /search?q=...&limit=...    movie title autocomplete
  POST /recommend                 full recommendation pipeline

Usage (local dev):
  uvicorn main:app --reload --port 8000

The recommend endpoint runs the full pipeline:
  1. recommender.get_candidates()  ─  FAISS semantic retrieval
  2. claude_reranker.rerank()      ─  LLM reranking + explanation

To develop without spending Claude API credits, swap rerank → rerank_mock
in the /recommend handler below.
"""

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from recommender import get_candidates, search_movies
from claude_reranker import rerank, rerank_mock

load_dotenv()

# ── APP ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Reelio API",
    description="ML-powered movie recommendation engine",
    version="0.1.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allows the React frontend (running on localhost:5173 or Vercel) to call this API.
# In production, replace "*" with your actual Vercel URL.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your Vercel URL before final deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """
    Request body for POST /recommend.

    seed_titles:  1–2 movie titles the user selected as seeds.
    preferences:  Dict of optional user preferences passed to Claude.
                  Supported keys:
                    genre       (str)  e.g. "Thriller"
                    mood        (str)  e.g. "something light"
                    runtime_max (int)  max minutes
                    group       (str)  e.g. "date night"
                    extra       (str)  any free text
    use_mock:     If true, skip the Claude API call (dev mode).
    """
    seed_titles: list[str] = Field(
        ...,
        min_length=1,
        max_length=2,
        description="1–2 seed movie titles",
        examples=[["Inception", "The Matrix"]],
    )
    preferences: dict = Field(
        default_factory=dict,
        description="Optional user preferences for Claude reranking",
        examples=[{"genre": "Sci-Fi", "runtime_max": 120, "group": "date night"}],
    )
    use_mock: bool = Field(
        default=False,
        description="Use mock reranker (no Claude API call). For development only.",
    )


class MovieResult(BaseModel):
    id:           int
    title:        str
    release_year: int | None
    genre_string: str
    runtime:      int | None
    vote_average: float | None
    poster_url:   str | None
    explanation:  str


class RecommendResponse(BaseModel):
    recommendations: list[MovieResult]
    resolved_seeds:  list[str]
    candidate_count: int


class SearchResult(BaseModel):
    id:           int
    title:        str
    release_year: int | None
    genre_string: str
    poster_url:   str | None


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Basic health check — confirms the server is running."""
    return {"status": "ok", "service": "Reelio API", "version": "0.1.0"}


@app.get("/search", response_model=list[SearchResult], tags=["Search"])
def search(
    q:     str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=8, ge=1, le=20, description="Max results"),
):
    """
    Autocomplete search against movie titles.
    Used by the frontend search bar to suggest known movies.

    Returns lightweight results (no embeddings, no Claude).
    Fast enough for live-as-you-type queries.
    """
    if not q.strip():
        return []

    try:
        results = search_movies(query=q, limit=limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommend"])
def recommend(body: RecommendRequest):
    """
    Full recommendation pipeline.

    1. Resolve seed titles → embedding vectors
    2. Average vectors → single FAISS query
    3. Retrieve top 25 semantic candidates
    4. Claude (or mock) reranks to top 5 with explanations

    Returns 5 recommended movies with explanations.
    """
    # ── Validate input ──
    cleaned_seeds = [t.strip() for t in body.seed_titles if t.strip()]
    if not cleaned_seeds:
        raise HTTPException(status_code=422, detail="At least one seed title is required.")

    # ── Step 1 + 2 + 3: FAISS retrieval ──
    try:
        candidates, resolved_seeds = get_candidates(
            seed_titles=cleaned_seeds,
            fallback_query=" ".join(cleaned_seeds),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No candidates found. Try different seed movies.",
        )

    # ── Step 4: Claude reranking ──
    reranker = rerank_mock if body.use_mock else rerank

    try:
        recommendations = reranker(
            candidates=candidates,
            resolved_seeds=resolved_seeds,
            preferences=body.preferences,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking error: {str(e)}")

    return RecommendResponse(
        recommendations=recommendations,
        resolved_seeds=resolved_seeds,
        candidate_count=len(candidates),
    )
