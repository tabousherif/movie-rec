"""
recommender.py
──────────────
Core retrieval layer for Reelio.

Responsibilities:
  1. Load the FAISS index + movie metadata on startup (once, into memory)
  2. Accept seed movie titles from the user
  3. Look up their embedding vectors, average them into a query vector
  4. Ask FAISS for the top-K nearest neighbors
  5. Return the candidate movies (to be reranked by Claude in claude_reranker.py)

This module is intentionally stateless beyond the loaded index — all
query logic is functional and side-effect free, making it easy to test.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── CONFIG ───────────────────────────────────────────────────────────────────

MODEL_NAME       = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = Path("data/movies.index")
MOVIE_INDEX_PATH = Path("data/movie_index.json")
EMBEDDINGS_PATH  = Path("data/embeddings.npy")

# Number of candidates to retrieve from FAISS before Claude reranks them.
# Higher = more diversity for Claude to work with; lower = faster.
FAISS_CANDIDATES = 25

# ── STARTUP: LOAD ARTIFACTS INTO MEMORY ──────────────────────────────────────
# These are loaded once when the module is first imported and reused across
# all requests. Loading on every request would be ~3s of overhead each time.

print("Loading ML artifacts into memory...")

_model = SentenceTransformer(MODEL_NAME)
print(f"  ✓ Sentence-transformer loaded  ({MODEL_NAME})")

_index = faiss.read_index(str(FAISS_INDEX_PATH))
print(f"  ✓ FAISS index loaded           ({_index.ntotal} vectors)")

with open(MOVIE_INDEX_PATH, "r", encoding="utf-8") as f:
    _movie_index: list[dict] = json.load(f)
print(f"  ✓ Movie index loaded           ({len(_movie_index)} movies)")

_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
print(f"  ✓ Embeddings loaded            (shape {_embeddings.shape})\n")

# Build a fast title → array_idx lookup (lowercased for fuzzy matching)
_title_lookup: dict[str, int] = {
    m["title"].lower(): m["array_idx"]
    for m in _movie_index
}

# ── HELPERS ──────────────────────────────────────────────────────────────────

def _find_movie_by_title(title: str) -> dict | None:
    """
    Find a movie in the index by title.
    Tries exact match first, then partial match.
    Returns the movie dict or None if not found.
    """
    lower = title.lower().strip()

    # 1. Exact match
    if lower in _title_lookup:
        idx = _title_lookup[lower]
        return _movie_index[idx]

    # 2. Partial match — find the first title that contains the query string
    for stored_title, idx in _title_lookup.items():
        if lower in stored_title:
            return _movie_index[idx]

    return None


def _get_embedding(array_idx: int) -> np.ndarray:
    """Return the embedding vector for a given array index. Shape: (384,)"""
    return _embeddings[array_idx].copy()


def _encode_freetext(text: str) -> np.ndarray:
    """
    Encode an arbitrary text string into a normalized embedding vector.
    Used when the user types something that isn't a known movie title
    (e.g. 'mind-bending sci-fi like Inception' or a genre description).
    """
    vec = _model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec[0].astype(np.float32)


def _build_query_vector(seed_movies: list[dict]) -> np.ndarray:
    """
    Average the embedding vectors of the seed movies into a single query vector,
    then re-normalize. Shape of output: (1, 384)

    Why average?
    If the user picks Inception + The Matrix, we want a query vector that sits
    between their two embedding positions in vector space — finding movies that
    share characteristics with both seeds rather than just one.
    """
    vecs = np.array([_get_embedding(m["array_idx"]) for m in seed_movies])
    averaged = vecs.mean(axis=0, keepdims=True).astype(np.float32)
    faiss.normalize_L2(averaged)   # re-normalize after averaging
    return averaged                # shape (1, 384)

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def search_movies(query: str, limit: int = 8) -> list[dict]:
    """
    Simple title/keyword search against the movie index.
    Used by the frontend search bar autocomplete endpoint.

    Returns a list of lightweight movie dicts (no embeddings).
    """
    lower = query.lower().strip()
    if not lower:
        return []

    results = []
    for movie in _movie_index:
        if lower in movie["title"].lower():
            results.append({
                "id":           movie["id"],
                "title":        movie["title"],
                "release_year": movie["release_year"],
                "genre_string": movie["genre_string"],
                "poster_url":   movie["poster_url"],
            })
        if len(results) >= limit:
            break

    return results


def get_candidates(
    seed_titles: list[str],
    fallback_query: str | None = None,
    k: int = FAISS_CANDIDATES,
) -> tuple[list[dict], list[str]]:
    """
    Core retrieval function.

    1. Resolve seed titles to known movies in the index.
    2. For unresolved titles, encode them as free-text and blend them in.
    3. Average all seed vectors into one query vector.
    4. Query FAISS for the k nearest neighbors.
    5. Return (candidates, resolved_titles).

    Args:
        seed_titles:    List of movie titles the user provided (1–2 titles).
        fallback_query: Optional free-text string to use if no seeds resolve
                        (e.g. the raw search query from the user).
        k:              Number of candidates to retrieve.

    Returns:
        candidates:      List of movie dicts from the FAISS results.
        resolved_titles: List of titles that were successfully found in the index.
    """
    seed_movies    = []
    freetext_vecs  = []
    resolved_titles = []
    unresolved     = []

    # ── Resolve each seed title ──
    for title in seed_titles:
        movie = _find_movie_by_title(title)
        if movie:
            seed_movies.append(movie)
            resolved_titles.append(movie["title"])
        else:
            unresolved.append(title)

    # ── Encode unresolved titles as free text ──
    for title in unresolved:
        vec = _encode_freetext(title)
        freetext_vecs.append(vec)

    # ── Build query vector ──
    if not seed_movies and not freetext_vecs:
        # Nothing resolved at all — fall back to raw query string
        if fallback_query:
            query_vec = _encode_freetext(fallback_query).reshape(1, -1).astype(np.float32)
        else:
            return [], []
    else:
        # Combine embedding vectors from resolved movies + freetext encodings
        all_vecs = (
            [_get_embedding(m["array_idx"]) for m in seed_movies]
            + freetext_vecs
        )
        averaged = np.array(all_vecs).mean(axis=0, keepdims=True).astype(np.float32)
        faiss.normalize_L2(averaged)
        query_vec = averaged

    # ── FAISS search ──
    scores, indices = _index.search(query_vec, k + len(seed_movies))
    # Retrieve k + seeds because we'll filter out the seeds from results

    # ── Build candidate list (exclude the seed movies themselves) ──
    seed_ids = {m["id"] for m in seed_movies}
    candidates = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(_movie_index):
            continue   # FAISS can return -1 for empty slots
        movie = _movie_index[idx]
        if movie["id"] in seed_ids:
            continue   # don't recommend what the user already told us they like
        candidates.append({
            **movie,
            "similarity_score": float(scores[0][rank]),
        })
        if len(candidates) >= k:
            break

    return candidates, resolved_titles
