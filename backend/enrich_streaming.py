"""
enrich_streaming.py
───────────────────
Enriches data/movie_index.json with Canadian streaming availability
from TMDB's watch providers endpoint.

What it adds to each movie:
  "streaming_ca": [
    {
      "provider_id": 8,
      "name":        "Netflix",
      "logo_url":    "https://image.tmdb.org/t/p/original/t2yyOv40HZeVlLjYsCsPHnWLk4W.jpg"
    },
    ...
  ]

Movies with no Canadian streaming availability get:
  "streaming_ca": []

Providers tracked (flatrate/subscription only — not rent or buy):
  Netflix, Amazon Prime Video, Disney+, HBO Max / Max, Apple TV+, Crave

Usage:
  python enrich_streaming.py

Runtime: ~20–35 min for 5,000 movies (TMDB rate limiting).
Safe to re-run — resumes from where it left off using a progress cache.

Output:
  data/movie_index.json    — updated in place
  data/streaming_cache.json — per-movie cache so re-runs are instant
"""

import json
import time
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────────────────────

TMDB_API_KEY   = os.getenv("TMDB_API_KEY")
BASE_URL       = "https://api.themoviedb.org/3"
LOGO_BASE      = "https://image.tmdb.org/t/p/original"
REGION         = "CA"
SLEEP_BETWEEN  = 0.26   # ~38 req/10s — safely under TMDB's 40/10s limit

MOVIE_INDEX_PATH   = Path("data/movie_index.json")
CACHE_PATH         = Path("data/streaming_cache.json")

# Provider ids we care about — flatrate (subscription) only
# Full list: https://www.themoviedb.org/talk/5e6f6f1b8a8e4a00138fc672
TRACKED_PROVIDERS = {
    8:   "Netflix",
    119: "Amazon Prime Video",
    337: "Disney+",
    384: "HBO Max",
    1899:"Max",          # HBO Max rebranded
    350: "Apple TV+",
    230: "Crave",        # Canadian exclusive
    531: "Paramount+",
}

# ── HELPERS ──────────────────────────────────────────────────────────────────

def fetch_providers(movie_id: int) -> list[dict]:
    """
    Fetch Canadian flatrate streaming providers for a single movie.
    Returns a list of provider dicts, or [] if none / request fails.
    """
    url    = f"{BASE_URL}/movie/{movie_id}/watch/providers"
    params = {"api_key": TMDB_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        ca_data   = data.get("results", {}).get(REGION, {})
        flatrate  = ca_data.get("flatrate", [])  # subscription providers only

        providers = []
        for p in flatrate:
            pid = p.get("provider_id")
            if pid not in TRACKED_PROVIDERS:
                continue
            logo_path = p.get("logo_path", "")
            providers.append({
                "provider_id": pid,
                "name":        TRACKED_PROVIDERS[pid],
                "logo_url":    f"{LOGO_BASE}{logo_path}" if logo_path else None,
            })

        return providers

    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return []   # movie not found — normal, return empty
        print(f"  HTTP error for id {movie_id}: {e}")
        return []
    except Exception as e:
        print(f"  Error for id {movie_id}: {e}")
        return []


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if not TMDB_API_KEY:
        raise ValueError("TMDB_API_KEY not set. Add it to your .env file.")

    # Load movie index
    if not MOVIE_INDEX_PATH.exists():
        raise FileNotFoundError(f"{MOVIE_INDEX_PATH} not found. Run fetch_movies.py first.")

    print(f"Loading {MOVIE_INDEX_PATH}...")
    with open(MOVIE_INDEX_PATH, "r", encoding="utf-8") as f:
        movies = json.load(f)
    print(f"  {len(movies)} movies loaded.\n")

    # Load progress cache (so re-runs skip already-fetched movies)
    cache: dict[str, list] = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Resuming — {len(cache)} movies already cached.\n")

    # ── Fetch loop ──
    newly_fetched = 0
    total         = len(movies)

    for i, movie in enumerate(movies):
        mid = str(movie["id"])

        if mid in cache:
            continue   # already fetched — skip

        providers = fetch_providers(movie["id"])
        cache[mid] = providers
        newly_fetched += 1

        # Progress every 100 movies
        if newly_fetched % 100 == 0:
            # Save cache checkpoint so we can resume if interrupted
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            pct = round((len(cache) / total) * 100)
            streaming_count = sum(1 for v in cache.values() if v)
            print(f"  {len(cache)}/{total} ({pct}%) — "
                  f"{streaming_count} have CA streaming  [checkpoint saved]")

        time.sleep(SLEEP_BETWEEN)

    # Save final cache
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    # ── Merge streaming data into movie_index ──
    print("\nMerging streaming data into movie_index.json...")

    provider_counts: dict[str, int] = {}
    has_streaming = 0

    for movie in movies:
        mid       = str(movie["id"])
        providers = cache.get(mid, [])
        movie["streaming_ca"] = providers

        if providers:
            has_streaming += 1
            for p in providers:
                name = p["name"]
                provider_counts[name] = provider_counts.get(name, 0) + 1

    # Save enriched movie_index
    with open(MOVIE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)

    # ── Summary ──
    print(f"\n✓ Done. {MOVIE_INDEX_PATH} updated.")
    print(f"\n── Canadian Streaming Coverage ───────────────────")
    print(f"  Movies with any CA streaming : {has_streaming} / {total} "
          f"({round(has_streaming/total*100)}%)")
    print(f"\n  By provider:")
    for name, count in sorted(provider_counts.items(), key=lambda x: -x[1]):
        print(f"    {name:<22} {count} movies")

    print(f"\n── Next steps ────────────────────────────────────")
    print(f"  1. Restart uvicorn — the enriched movie_index will be loaded automatically")
    print(f"  2. streaming_ca is now on every movie dict that flows through get_candidates()")
    print(f"  3. The frontend just needs to read movie.streaming_ca from the API response")


if __name__ == "__main__":
    main()
