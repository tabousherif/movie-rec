"""
build_index.py
──────────────
Loads the embedding matrix produced by embed_movies.py and builds
a FAISS index for fast nearest-neighbor search at query time.

What it produces:
  data/movies.index  ─  FAISS flat inner-product index, ready to query

Usage:
  python build_index.py

Why FAISS?
  With 5,000 movies a brute-force dot product is fast enough, but FAISS
  gives us a standard interface that scales to millions of vectors with no
  code changes (swap IndexFlatIP for IndexIVFFlat or HNSW when needed).

Index type: IndexFlatIP (exact inner product search)
  - "Flat"  = exhaustive, exact search (no approximation)
  - "IP"    = inner product (== cosine similarity for L2-normalized vecs)
  - Correct for datasets up to ~100k; swap to approximate index beyond that
"""

import json
import numpy as np
import faiss
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────

EMBEDDINGS_PATH  = Path("data/embeddings.npy")
MOVIE_INDEX_PATH = Path("data/movie_index.json")
FAISS_INDEX_PATH = Path("data/movies.index")

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():

    # ── 1. Load embeddings ────────────────────────────────────────────────────
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_PATH} not found.\n"
            "Run embed_movies.py first."
        )

    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"  Shape : {embeddings.shape}")
    print(f"  dtype : {embeddings.dtype}\n")

    # FAISS requires float32
    if embeddings.dtype != np.float32:
        print("  Converting to float32...")
        embeddings = embeddings.astype(np.float32)

    n_movies, dim = embeddings.shape

    # ── 2. Build FAISS index ──────────────────────────────────────────────────
    print(f"Building FAISS IndexFlatIP  (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"  Vectors added : {index.ntotal}")
    assert index.ntotal == n_movies, "Vector count mismatch — re-run from scratch."

    # ── 3. Save to disk ───────────────────────────────────────────────────────
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"\n✓ FAISS index saved  →  {FAISS_INDEX_PATH}")

    # ── 4. Sanity check ───────────────────────────────────────────────────────
    if not MOVIE_INDEX_PATH.exists():
        print("\nWarning: movie_index.json not found — skipping sanity check.")
        return

    with open(MOVIE_INDEX_PATH, "r", encoding="utf-8") as f:
        movie_index = json.load(f)

    print("\n── Sanity Check: Live FAISS Query ────────────────────────────────")

    # Average the first two movies' vectors to form a blended query
    seed_movies = movie_index[:2]
    seed_vecs   = embeddings[:2].copy()
    query_vec   = seed_vecs.mean(axis=0, keepdims=True).astype(np.float32)
    faiss.normalize_L2(query_vec)   # re-normalize after averaging

    k = 7
    scores, indices = index.search(query_vec, k)

    seed_idx_set = {0, 1}
    results = [
        (movie_index[idx], float(scores[0][rank]))
        for rank, idx in enumerate(indices[0])
        if idx not in seed_idx_set
    ][:5]

    print(f"  Seed 1 : {seed_movies[0]['title']}")
    print(f"  Seed 2 : {seed_movies[1]['title']}")
    print(f"\n  Top 5 FAISS results for blended query:")
    for movie, score in results:
        print(f"    • {movie['title']} ({movie.get('release_year', '?')})  "
              f"[{movie['genre_string']}]  score={score:.4f}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n── Pipeline Complete ─────────────────────────────────────────────")
    print(f"  data/movies.json      ─  {len(movie_index)} movies (metadata)")
    print(f"  data/embeddings.npy   ─  shape {embeddings.shape} (float32)")
    print(f"  data/movies.index     ─  FAISS index, {index.ntotal} vectors")
    print(f"  data/movie_index.json ─  position → movie lookup table")
    print("\n  All three ML artifacts are ready. ✓")


if __name__ == "__main__":
    main()
