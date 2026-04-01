"""
claude_reranker.py
──────────────────
Reranks FAISS candidates using Claude.

Claude returns (id, explanation, score 1-100) for exactly 5 picks.
All other metadata is hydrated from the candidate lookup.
If Claude returns fewer than 5, we pad with the next-best FAISS candidates.
Duplicate ids are deduplicated at the output stage.
"""

import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL  = "claude-sonnet-4-20250514"
TARGET = 5   # always return exactly this many recommendations

# ── PROMPTS ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are Reelio's recommendation engine — a film-literate assistant.

You will receive a numbered list of candidate movies and the user's preferences.

Your task:
- Select exactly {TARGET} distinct movies from the candidates
- For each, write a 2-3 sentence explanation referencing tone, themes, or style
- Assign each a recommendation score from 1-100 reflecting how well it fits
  the user's preferences (not the film's general quality). The best fit = 100.
- Order your picks from highest to lowest score.

Output ONLY a valid JSON array. No markdown, no code fences, no extra text.
Each element must have exactly three keys:

  "id"          — integer, copied exactly from the candidate list
  "score"       — integer 1-100, your recommendation strength score
  "explanation" — string, 2-3 sentences on why this film fits

Example output:
[
  {{"id": 27205, "score": 96, "explanation": "Cerebral and visually inventive. Shares the layered reality structure of your picks."}},
  {{"id": 157336, "score": 88, "explanation": "Epic, emotionally grounded sci-fi. Matches the ambitious scope of your selections."}}
]"""


def _build_prompt(candidates: list[dict], resolved_seeds: list[str], preferences: dict) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        runtime = f"{c['runtime']}min" if c.get("runtime") else "?"
        genre   = c.get("genre_string") or "Unknown"
        rating  = c.get("vote_average") or "?"
        year    = c.get("release_year") or "?"
        lines.append(
            f"{i:>2}. [id={c['id']}] {c['title']} ({year}) "
            f"| {genre} | {runtime} | TMDB {rating}/10"
        )

    prefs = []
    if resolved_seeds:
        prefs.append(f"Seed movies the user enjoyed: {', '.join(resolved_seeds)}")
    if preferences.get("genre"):
        prefs.append(f"Preferred genre: {preferences['genre']}")
    if preferences.get("mood"):
        prefs.append(f"Mood/vibe: {preferences['mood']}")
    if preferences.get("runtime_max"):
        prefs.append(f"Max runtime: {preferences['runtime_max']} minutes")
    if preferences.get("group"):
        prefs.append(f"Watching with: {preferences['group']}")

    prefs_block = "\n".join(prefs) or "No specific preferences stated."
    cands_block = "\n".join(lines)

    return (
        f"USER PREFERENCES\n{prefs_block}\n\n"
        f"CANDIDATES ({len(candidates)} films)\n{cands_block}\n\n"
        f"Pick exactly {TARGET} and return JSON."
    )


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _hydrate(pick: dict, candidate_by_id: dict[int, dict]) -> dict | None:
    """Merge Claude's (id, score, explanation) with full metadata from candidate lookup."""
    pid = pick.get("id")
    if pid is None:
        return None
    source = candidate_by_id.get(int(pid))
    if source is None:
        print(f"  [reranker] id {pid} not in candidates — skipping")
        return None
    return {
        **source,
        "rec_score":   int(pick.get("score", source.get("rec_score", 70))),
        "explanation": str(pick.get("explanation", "")),
    }


def _pad_to_target(results: list[dict], candidates: list[dict], seen_ids: set) -> list[dict]:
    """
    If Claude returned fewer than TARGET picks, fill with next-best FAISS candidates.
    Preserves descending rec_score order and never adds duplicates.
    """
    for c in candidates:
        if len(results) >= TARGET:
            break
        if c["id"] in seen_ids:
            continue
        m = dict(c)
        m.setdefault("explanation", "Strong semantic match to your selections.")
        results.append(m)
        seen_ids.add(c["id"])

    return results


def _sort_and_cap(results: list[dict]) -> list[dict]:
    """Sort by rec_score descending, deduplicate by id, cap at TARGET."""
    seen    = set()
    unique  = []
    for r in sorted(results, key=lambda x: x.get("rec_score", 0), reverse=True):
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
        if len(unique) >= TARGET:
            break
    return unique


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def rerank(candidates: list[dict], resolved_seeds: list[str], preferences: dict) -> list[dict]:
    """
    Ask Claude to pick, score, and explain top recommendations.
    Always returns exactly TARGET (5) results, padded from FAISS if needed.
    Never contains duplicate movie ids.
    """
    if not candidates:
        return []

    # Immutable copies so we never mutate the shared _movie_index entries
    candidate_by_id = {c["id"]: dict(c) for c in candidates}

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_prompt(candidates, resolved_seeds, preferences)}],
        )

        raw = response.content[0].text.strip()

        # Strip accidental markdown fences
        if "```" in raw:
            parts = raw.split("```")
            raw   = parts[1] if len(parts) > 1 else parts[0]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        picks = json.loads(raw)

        if not isinstance(picks, list):
            raise ValueError(f"Expected list, got {type(picks)}")

        # Hydrate picks with full metadata
        results  = []
        seen_ids = set()
        for pick in picks:
            hydrated = _hydrate(pick, candidate_by_id)
            if hydrated and hydrated["id"] not in seen_ids:
                results.append(hydrated)
                seen_ids.add(hydrated["id"])

        print(f"  [reranker] Claude returned {len(results)} valid picks")

        # Always guarantee exactly TARGET results
        results = _pad_to_target(results, candidates, seen_ids)
        results = _sort_and_cap(results)

        print(f"  [reranker] final count: {len(results)}, "
              f"scores: {[r['rec_score'] for r in results]}")
        return results

    except Exception as e:
        print(f"  [reranker] ERROR: {e} — using FAISS fallback")
        return _fallback(candidates)


def rerank_mock(candidates: list[dict], **_) -> list[dict]:
    """
    Dev mock — returns exactly TARGET results with placeholder explanations.
    Deduplicates and sorts by rec_score. Never mutates input.
    """
    seen    = set()
    results = []
    for c in candidates:
        if len(results) >= TARGET:
            break
        if c["id"] in seen:
            continue
        m = dict(c)
        m["explanation"] = (
            f"'{m['title']}' shares strong thematic similarities with your picks. "
            f"[{m.get('genre_string', 'Unknown genre')}]"
        )
        results.append(m)
        seen.add(m["id"])

    return _sort_and_cap(results)


def _fallback(candidates: list[dict]) -> list[dict]:
    """Return exactly TARGET results from FAISS order. Deduplicates. Never mutates."""
    seen    = set()
    results = []
    for c in candidates:
        if len(results) >= TARGET:
            break
        if c["id"] in seen:
            continue
        m = dict(c)
        m.setdefault("explanation", "Strong semantic match to your selections.")
        results.append(m)
        seen.add(m["id"])
    return _sort_and_cap(results)
