"""
main.py
───────
Reelio FastAPI backend.

Endpoints:
  GET  /health                    health + index stats
  GET  /search?q=...&limit=...    movie title autocomplete
  POST /recommend                 full recommendation pipeline

Usage:
  uvicorn main:app --reload --port 8000
"""

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from recommender import get_candidates, search_movies
from claude_reranker import rerank, rerank_mock

load_dotenv()

app = FastAPI(title="Reelio API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODELS ────────────────────────────────────────────────────────────────────

class SeedMovie(BaseModel):
    id:    int
    title: str


class RecommendRequest(BaseModel):
    seeds:       list[SeedMovie] = Field(..., min_length=1, max_length=2)
    preferences: dict            = Field(default_factory=dict)
    use_mock:    bool            = Field(default=False)


class StreamingProvider(BaseModel):
    provider_id: int
    name:        str
    logo_url:    str | None = None


class MovieResult(BaseModel):
    id:            int
    title:         str
    release_year:  int   | None = None
    genre_string:  str   | None = None
    runtime:       int   | None = None
    vote_average:  float | None = None
    poster_url:    str   | None = None
    explanation:   str   | None = None
    rec_score:     int   | None = None
    streaming_ca:  list[StreamingProvider] = []


class RecommendResponse(BaseModel):
    recommendations: list[MovieResult]
    resolved_seeds:  list[str]
    candidate_count: int


class SearchResult(BaseModel):
    id:           int
    title:        str
    release_year: int   | None = None
    genre_string: str   | None = None
    poster_url:   str   | None = None


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/")
def health():
    return {"status": "ok", "service": "Reelio API", "version": "0.1.0"}


@app.get("/search", response_model=list[SearchResult])
def search(
    q:     str = Query(..., min_length=1),
    limit: int = Query(default=8, ge=1, le=20),
):
    try:
        return search_movies(query=q.strip(), limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest):
    if not body.seeds:
        raise HTTPException(status_code=422, detail="At least one seed movie is required.")

    seed_ids    = [s.id    for s in body.seeds]
    seed_titles = [s.title for s in body.seeds]

    print(f"\n[/recommend] seeds={list(zip(seed_ids, seed_titles))} mock={body.use_mock}")

    # Step 1-3: FAISS retrieval
    try:
        candidates, resolved_seeds = get_candidates(
            seed_titles=seed_titles,
            seed_ids=seed_ids,
            fallback_query=" ".join(seed_titles),
        )
    except Exception as e:
        print(f"[/recommend] retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    print(f"[/recommend] candidates={len(candidates)} resolved={resolved_seeds}")

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No candidates found for seeds: {seed_titles}. "
                "These titles may not be in the dataset. Try a different movie."
            ),
        )

    # Step 4: reranking
    reranker = rerank_mock if body.use_mock else rerank
    try:
        recommendations = reranker(
            candidates=candidates,
            resolved_seeds=resolved_seeds,
            preferences=body.preferences,
        )
    except Exception as e:
        print(f"[/recommend] reranker error: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking error: {e}")

    print(f"[/recommend] returning {len(recommendations)} recommendations")

    return RecommendResponse(
        recommendations=recommendations,
        resolved_seeds=resolved_seeds,
        candidate_count=len(candidates),
    )