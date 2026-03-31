"""
fetch_movies.py
───────────────
Pulls movie data from the TMDB API and saves it as movies.json.

What it fetches per movie:
  - id, title, overview (description), genres, release_year,
    runtime, vote_average, vote_count, popularity, poster_path

Usage:
  1. Set your TMDB API key in a .env file:
       TMDB_API_KEY=your_key_here
  2. Run:
       python fetch_movies.py

Output:
  data/movies.json  ─  list of movie dicts, ready for embedding

Notes:
  - Fetches from the "discover" endpoint sorted by popularity
  - Skips movies with missing overviews (essential for embeddings)
  - Respects TMDB rate limits with a small sleep between requests
  - Running this once is enough; output is saved and reused
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────

TMDB_API_KEY  = os.getenv("TMDB_API_KEY")
BASE_URL      = "https://api.themoviedb.org/3"
POSTER_BASE   = "https://image.tmdb.org/t/p/w500"  # poster URL prefix
TARGET_MOVIES = 5000   # how many movies to fetch total
PAGES_NEEDED  = TARGET_MOVIES // 20  # TMDB returns 20 movies per page (max 250 pages)
SLEEP_BETWEEN = 0.25   # seconds between requests (TMDB rate limit: 40 req/10s)
OUTPUT_PATH   = Path("data/movies.json")

# ── GENRE MAP ────────────────────────────────────────────────────────────────
# TMDB returns genre IDs — we resolve them to names upfront
GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

# ── HELPERS ──────────────────────────────────────────────────────────────────

def get_genre_names(genre_ids: list[int]) -> list[str]:
    return [GENRE_MAP[gid] for gid in genre_ids if gid in GENRE_MAP]


def fetch_page(page: int) -> list[dict]:
    """Fetch one page of popular movies from TMDB discover endpoint."""
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "include_adult": False,
        "include_video": False,
        "page": page,
        "vote_count.gte": 100,   # filter out obscure films with few votes
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("results", [])


def fetch_runtime(movie_id: int) -> int | None:
    """Fetch runtime for a single movie (not in discover endpoint)."""
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        return resp.json().get("runtime")  # in minutes, can be None
    except Exception:
        return None


def clean_movie(raw: dict, runtime: int | None) -> dict | None:
    """
    Transform a raw TMDB result into our clean movie dict.
    Returns None if the movie is missing an overview (unusable for embeddings).
    """
    overview = raw.get("overview", "").strip()
    if not overview or len(overview) < 30:
        return None  # skip movies without meaningful descriptions

    genres = get_genre_names(raw.get("genre_ids", []))
    release_year = int(raw["release_date"][:4]) if raw.get("release_date") else None
    poster_path  = raw.get("poster_path")

    return {
        "id":           raw["id"],
        "title":        raw.get("title", "").strip(),
        "overview":     overview,
        "genres":       genres,
        "genre_string": ", ".join(genres),   # flat string for embedding
        "release_year": release_year,
        "runtime":      runtime,             # minutes (may be None)
        "vote_average": round(raw.get("vote_average", 0), 1),
        "vote_count":   raw.get("vote_count", 0),
        "popularity":   round(raw.get("popularity", 0), 2),
        "poster_url":   f"{POSTER_BASE}{poster_path}" if poster_path else None,
        # This is what gets embedded — rich text combining title + genres + overview
        "embed_text":   f"{raw.get('title', '')}. {', '.join(genres)}. {overview}",
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if not TMDB_API_KEY:
        raise ValueError(
            "TMDB_API_KEY not found. "
            "Add it to a .env file in this directory:\n"
            "  TMDB_API_KEY=your_key_here"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    movies     = []
    seen_ids   = set()
    skipped    = 0
    page_count = min(PAGES_NEEDED, 250)  # TMDB caps discover at 500 pages

    print(f"Fetching up to {TARGET_MOVIES} movies across {page_count} pages...")
    print("This will take a few minutes — TMDB rate limits require small delays.\n")

    for page in range(1, page_count + 1):
        try:
            results = fetch_page(page)
        except requests.HTTPError as e:
            print(f"  ✗ Page {page} failed: {e} — skipping")
            time.sleep(2)
            continue

        for raw in results:
            movie_id = raw.get("id")
            if not movie_id or movie_id in seen_ids:
                continue

            # Fetch runtime (separate API call — we batch these carefully)
            runtime = fetch_runtime(movie_id)
            time.sleep(SLEEP_BETWEEN)

            movie = clean_movie(raw, runtime)
            if movie is None:
                skipped += 1
                continue

            movies.append(movie)
            seen_ids.add(movie_id)

        # Progress update every 10 pages
        if page % 10 == 0:
            print(f"  Page {page}/{page_count} — {len(movies)} movies collected so far")

        time.sleep(SLEEP_BETWEEN)

        if len(movies) >= TARGET_MOVIES:
            print(f"\nReached target of {TARGET_MOVIES} movies.")
            break

    # ── SAVE ──
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Done.")
    print(f"  Saved  : {OUTPUT_PATH}")
    print(f"  Movies : {len(movies)}")
    print(f"  Skipped: {skipped} (missing overview)")
    print(f"\nNext step: run  python embed_movies.py")


if __name__ == "__main__":
    main()
