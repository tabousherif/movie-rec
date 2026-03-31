"""
embed_movies.py
───────────────
Loads movies.json, encodes each movie's embed_text field using a
sentence-transformer model, and saves the resulting embedding matrix.

What it produces:
  data/embeddings.npy    ─  float32 matrix, shape (N, 384)
                             one 384-dimensional vector per movie
  data/movie_index.json  ─  ordered list of movie metadata, where
                             position i corresponds to embedding row i

Usage:
  python embed_movies.py

Notes:
  - Uses 'all-MiniLM-L6-v2': fast, lightweight, strong semantic quality
  - First run downloads the model (~90MB) — cached locally after that
  - Encodes in batches of 64 for memory efficiency
  - L2-normalizes embeddings so cosine similarity = dot product (faster)
  - Runs a quick similarity sanity check at the end so you can verify
    the embeddings make intuitive sense before building the FAISS index
  - Runtime: ~2–4 min for 5,000 movies on a standard laptop CPU
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── CONFIG ───────────────────────────────────────────────────────────────────

MODEL_NAME       = "all-MiniLM-L6-v2"   # 384-dim, ~90MB, excellent quality/speed
BATCH_SIZE       = 64                    # encode 64 movies at a time
MOVIES_PATH      = Path("data/movies.json")
EMBEDDINGS_PATH  = Path("data/embeddings.npy")
MOVIE_INDEX_PATH = Path("data/movie_index.json")

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():

    # ── 1. Load movies ────────────────────────────────────────────────────────
    if not MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"{MOVIES_PATH} not found.\n"
            "Run fetch_movies.py first to generate the movie dataset."
        )

    print(f"Loading movies from {MOVIES_PATH}...")
    with open(MOVIES_PATH, "r", encoding="utf-8") as f:
        movies = json.load(f)

    print(f"  Loaded {len(movies)} movies.\n")

    # ── 2. Extract embed texts ─────────────────────────────────────────────────
    # embed_text = "Title. Genre1, Genre2. Full overview description."
    #
    # Why this format?
    # The sentence-transformer sees a rich, natural-language description of each
    # movie. The genre prefix gives explicit semantic anchors (e.g. "Thriller")
    # while the overview provides nuanced context about tone, plot, and themes.
    # This outperforms embedding title-only or overview-only.
    texts = [m["embed_text"] for m in movies]

    # ── 3. Load model ──────────────────────────────────────────────────────────
    print(f"Loading sentence-transformer model: '{MODEL_NAME}'")
    print("  First run: downloads ~90MB model weights (cached after that).\n")
    model = SentenceTransformer(MODEL_NAME)

    # ── 4. Encode in batches ───────────────────────────────────────────────────
    print(f"Encoding {len(texts)} movies in batches of {BATCH_SIZE}...")
    print("  Expected time: 2–4 min on CPU. Grab a coffee.\n")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize: cosine sim becomes a dot product
    )
    # embeddings.shape → (N, 384)  e.g. (5000, 384)

    print(f"\n  Shape : {embeddings.shape}")
    print(f"  dtype : {embeddings.dtype}")

    # ── 5. Save embedding matrix ───────────────────────────────────────────────
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"\n✓ Embeddings saved  →  {EMBEDDINGS_PATH}")

    # ── 6. Save movie index ────────────────────────────────────────────────────
    # FAISS returns integer indices (0, 1, 2, ...) into the embedding matrix.
    # movie_index.json lets us resolve those integers back to real movie data.
    # Critical: the order here must exactly match the embedding row order.
    movie_index = [
        {
            "array_idx":    i,
            "id":           m["id"],
            "title":        m["title"],
            "overview":     m["overview"],
            "genres":       m["genres"],
            "genre_string": m["genre_string"],
            "release_year": m["release_year"],
            "runtime":      m["runtime"],
            "vote_average": m["vote_average"],
            "vote_count":   m["vote_count"],
            "popularity":   m["popularity"],
            "poster_url":   m["poster_url"],
            "embed_text":   m["embed_text"],
        }
        for i, m in enumerate(movies)
    ]

    with open(MOVIE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(movie_index, f, ensure_ascii=False, indent=2)

    print(f"✓ Movie index saved →  {MOVIE_INDEX_PATH}")

    # ── 7. Sanity check ────────────────────────────────────────────────────────
    # Verify counts match, then run a quick nearest-neighbor test.
    # If the embeddings are working correctly, the top results for any seed movie
    # should feel intuitively similar — same genre, tone, or themes.
    print("\n── Sanity Check ──────────────────────────────────────────────────")
    print(f"  Movies in dataset  : {len(movies)}")
    print(f"  Embedding rows     : {embeddings.shape[0]}")
    print(f"  Embedding dims     : {embeddings.shape[1]}")
    assert len(movies) == embeddings.shape[0], "Mismatch! Re-run from scratch."
    print(f"  Row count matches  : ✓")

    print("\n── Quick Similarity Test ─────────────────────────────────────────")
    # Pick the first movie as a seed, find 3 nearest neighbors via dot product
    sample      = movies[0]
    sample_vec  = embeddings[0]                     # shape (384,)
    sims        = embeddings @ sample_vec           # shape (N,) — dot products
    top_indices = np.argsort(sims)[::-1][1:4]       # top 3, skipping itself

    print(f"  Seed   : {sample['title']} ({sample.get('release_year', '?')})")
    print(f"  Genres : {sample['genre_string']}")
    print(f"\n  Most similar movies found:")
    for idx in top_indices:
        m    = movies[idx]
        sim  = sims[idx]
        print(f"    • {m['title']} ({m.get('release_year', '?')})  "
              f"[{m['genre_string']}]  sim={sim:.4f}")

    print("\n  If these results feel thematically similar to the seed, ")
    print("  your embeddings are working correctly. ✓")
    print("\n── Next step: python build_index.py ──────────────────────────────")


if __name__ == "__main__":
    main()
