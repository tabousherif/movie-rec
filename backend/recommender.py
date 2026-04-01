"""
recommender.py
──────────────
Core ML retrieval layer. Loads artifacts once at startup, exposes
search_movies() and get_candidates() for use by main.py.
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
FAISS_CANDIDATES = 25

# ── STARTUP ──────────────────────────────────────────────────────────────────

print("Loading ML artifacts into memory...")

_model = SentenceTransformer(MODEL_NAME)
print(f"  ✓ Sentence-transformer  ({MODEL_NAME})")

_index = faiss.read_index(str(FAISS_INDEX_PATH))
print(f"  ✓ FAISS index           ({_index.ntotal} vectors)")

with open(MOVIE_INDEX_PATH, "r", encoding="utf-8") as f:
    _movie_index: list[dict] = json.load(f)
print(f"  ✓ Movie index           ({len(_movie_index)} movies)")

_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
print(f"  ✓ Embeddings            (shape {_embeddings.shape})\n")

# Lookup tables
_title_lookup: dict[str, int] = {
    m["title"].lower(): m["array_idx"]
    for m in _movie_index
}
_id_lookup: dict[int, int] = {
    m["id"]: m["array_idx"]
    for m in _movie_index
}

# ── HELPERS ──────────────────────────────────────────────────────────────────

def _safe_str(val, fallback: str = "") -> str:
    """Return val as a string, or fallback if None/empty."""
    if val is None:
        return fallback
    s = str(val).strip()
    return s if s else fallback


def _find_by_id(tmdb_id: int) -> dict | None:
    idx = _id_lookup.get(tmdb_id)
    return _movie_index[idx] if idx is not None else None


def _find_by_title(title: str) -> dict | None:
    lower = title.lower().strip()
    # 1. Exact
    idx = _title_lookup.get(lower)
    if idx is not None:
        return _movie_index[idx]
    # 2. Partial
    for stored, idx in _title_lookup.items():
        if lower in stored:
            return _movie_index[idx]
    return None


def _get_vec(array_idx: int) -> np.ndarray:
    return _embeddings[array_idx].copy()


def _encode(text: str) -> np.ndarray:
    vec = _model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec[0].astype(np.float32)


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def search_movies(query: str, limit: int = 8) -> list[dict]:
    """Title substring search for autocomplete."""
    lower = query.lower().strip()
    if not lower:
        return []
    results = []
    for m in _movie_index:
        if lower in m["title"].lower():
            results.append({
                "id":           m["id"],
                "title":        m["title"],
                "release_year": m.get("release_year"),
                "genre_string": _safe_str(m.get("genre_string")),
                "poster_url":   m.get("poster_url"),
            })
            if len(results) >= limit:
                break
    return results


def get_candidates(
    seed_titles: list[str],
    seed_ids: list[int] | None = None,
    fallback_query: str | None = None,
    k: int = FAISS_CANDIDATES,
) -> tuple[list[dict], list[str]]:
    """
    Resolve seeds → embedding vectors → FAISS search → candidate list.
    Returns (candidates, resolved_titles).
    """
    seed_movies:     list[dict]      = []
    freetext_vecs:   list[np.ndarray] = []
    resolved_titles: list[str]       = []

    ids        = seed_ids or []
    padded_ids = ids + [None] * max(0, len(seed_titles) - len(ids))

    print(f"[get_candidates] seeds received: {list(zip(padded_ids, seed_titles))}")

    for tmdb_id, title in zip(padded_ids, seed_titles):
        movie = None

        # 1. id-based lookup (exact, fast)
        if tmdb_id is not None:
            movie = _find_by_id(int(tmdb_id))
            if movie:
                print(f"  ✓ id lookup  {tmdb_id} → '{movie['title']}'")
            else:
                print(f"  ✗ id {tmdb_id} not in index — trying title fallback")

        # 2. title-based fallback
        if movie is None:
            movie = _find_by_title(title)
            if movie:
                print(f"  ✓ title lookup '{title}' → '{movie['title']}'")
            else:
                print(f"  ✗ title '{title}' not matched — using freetext encode")

        if movie:
            seed_movies.append(movie)
            resolved_titles.append(movie["title"])
        else:
            freetext_vecs.append(_encode(title))

    print(f"  resolved: {resolved_titles}, freetext_vecs: {len(freetext_vecs)}")

    # Build query vector
    if not seed_movies and not freetext_vecs:
        if fallback_query:
            print(f"  using fallback_query: '{fallback_query}'")
            query_vec = _encode(fallback_query).reshape(1, -1)
        else:
            print("  ERROR: nothing to query with")
            return [], []
    else:
        all_vecs = [_get_vec(m["array_idx"]) for m in seed_movies] + freetext_vecs
        averaged = np.array(all_vecs, dtype=np.float32).mean(axis=0, keepdims=True)
        faiss.normalize_L2(averaged)
        query_vec = averaged

    # FAISS search — retrieve extra so we can filter seeds out
    fetch_k = k + len(seed_movies) + 5
    scores, indices = _index.search(query_vec.astype(np.float32), fetch_k)

    seed_id_set = {m["id"] for m in seed_movies}
    candidates  = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(_movie_index):
            continue
        m = _movie_index[idx]
        if m["id"] in seed_id_set:
            continue
        candidates.append({
            **m,
            # Guarantee these fields are always safe strings, never None
            "genre_string": _safe_str(m.get("genre_string")),
            "title":        _safe_str(m.get("title"), fallback="Unknown"),
            "similarity_score": float(scores[0][rank]),
        })
        if len(candidates) >= k:
            break

    print(f"  FAISS returned {len(candidates)} candidates")
    return candidates, resolved_titles
