"""
claude_reranker.py
──────────────────
Takes the 25 FAISS candidates from recommender.py and uses Claude to:
  1. Filter out poor fits given the user's stated preferences
  2. Rank the remaining candidates by how well they match
  3. Return the top 5 with a human-readable explanation for each

Why LLM reranking?
  FAISS finds movies that are semantically close in vector space. But it
  can't reason about user preferences like "I only have 90 minutes" or
  "watching with my parents — nothing violent." Claude bridges that gap.
  This is the RAG pattern: retrieve broadly with ML, refine with LLM.

Prompt design:
  - Structured candidate list so Claude can reason over concrete options
  - User preferences injected as natural language constraints
  - Output requested as strict JSON so we can parse it reliably
  - System prompt keeps Claude focused and output format consistent
"""

import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL  = "claude-sonnet-4-20250514"

# ── PROMPT TEMPLATES ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Reelio's recommendation engine — a film-literate assistant
that selects and explains movie recommendations.

You will receive:
1. A list of candidate movies retrieved by a semantic ML pipeline
2. The user's stated preferences (runtime, mood, group, genre, etc.)

Your task:
- Select the 5 best matches from the candidates given the user's preferences
- For each pick, write a SHORT (2–3 sentence) explanation of why it fits
- Be specific: reference the film's tone, themes, or style — not just genre

Output ONLY a JSON array. No preamble, no markdown, no extra text.
Each element must have exactly these keys:
  id, title, release_year, genre_string, runtime, vote_average,
  poster_url, explanation

Example element:
{
  "id": 27205,
  "title": "Inception",
  "release_year": 2010,
  "genre_string": "Action, Science Fiction, Adventure",
  "runtime": 148,
  "vote_average": 8.4,
  "poster_url": "https://image.tmdb.org/t/p/w500/...",
  "explanation": "A layered, cerebral thriller that rewards close attention.
  If you liked the mind-bending structure of your seed movies, Inception
  delivers that same sense of reality shifting under your feet."
}"""


def _build_user_prompt(
    candidates: list[dict],
    resolved_seeds: list[str],
    preferences: dict,
) -> str:
    """
    Build the user-turn prompt containing candidate list + preferences.

    preferences keys (all optional):
      genre       str   e.g. "Thriller"
      mood        str   e.g. "something light and funny"
      runtime_max int   maximum runtime in minutes
      group       str   e.g. "date night", "watching alone", "with kids"
      extra       str   any free-text preference from the user
    """
    # ── Format candidate list ──
    candidate_lines = []
    for i, c in enumerate(candidates, 1):
        runtime_str = f"{c['runtime']}min" if c.get("runtime") else "runtime unknown"
        candidate_lines.append(
            f"{i:>2}. {c['title']} ({c.get('release_year', '?')}) "
            f"| {c['genre_string']} "
            f"| {runtime_str} "
            f"| rating {c.get('vote_average', '?')}/10"
        )
    candidates_block = "\n".join(candidate_lines)

    # ── Format preferences ──
    pref_lines = []
    if resolved_seeds:
        pref_lines.append(f"Seed movies they enjoyed: {', '.join(resolved_seeds)}")
    if preferences.get("genre"):
        pref_lines.append(f"Genre mood: {preferences['genre']}")
    if preferences.get("mood"):
        pref_lines.append(f"Vibe they're after: {preferences['mood']}")
    if preferences.get("runtime_max"):
        pref_lines.append(f"Maximum runtime: {preferences['runtime_max']} minutes")
    if preferences.get("group"):
        pref_lines.append(f"Watching with: {preferences['group']}")
    if preferences.get("extra"):
        pref_lines.append(f"Other notes: {preferences['extra']}")

    prefs_block = "\n".join(pref_lines) if pref_lines else "No specific preferences stated."

    return f"""USER PREFERENCES
{prefs_block}

CANDIDATE MOVIES ({len(candidates)} retrieved by semantic ML pipeline)
{candidates_block}

Select the 5 best matches and return JSON as specified."""


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def rerank(
    candidates: list[dict],
    resolved_seeds: list[str],
    preferences: dict,
) -> list[dict]:
    """
    Send candidates + preferences to Claude and return top 5 ranked movies.

    Args:
        candidates:     List of movie dicts from recommender.get_candidates()
        resolved_seeds: Movie titles that were matched in the FAISS index
        preferences:    Dict of user preferences (see _build_user_prompt docstring)

    Returns:
        List of up to 5 movie dicts, each with an 'explanation' field added.
        Falls back to top 5 FAISS candidates (no explanation) if Claude fails.
    """
    if not candidates:
        return []

    user_prompt = _build_user_prompt(candidates, resolved_seeds, preferences)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences if Claude wraps output (defensive)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        recommendations = json.loads(raw)

        # Validate structure — ensure each item has required keys
        required = {"id", "title", "explanation"}
        for item in recommendations:
            if not required.issubset(item.keys()):
                raise ValueError(f"Missing keys in Claude response item: {item}")

        return recommendations[:5]

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Graceful fallback: return top 5 FAISS candidates without explanation
        print(f"Claude reranker parse error: {e} — falling back to FAISS top 5")
        fallback = candidates[:5]
        for m in fallback:
            m.setdefault("explanation", "Recommended based on semantic similarity to your selections.")
        return fallback

    except anthropic.APIError as e:
        # API-level error (rate limit, auth, etc.)
        print(f"Claude API error: {e} — falling back to FAISS top 5")
        fallback = candidates[:5]
        for m in fallback:
            m.setdefault("explanation", "Recommended based on semantic similarity to your selections.")
        return fallback


def rerank_mock(candidates: list[dict], **_) -> list[dict]:
    """
    Drop-in mock for rerank() — returns top 5 FAISS candidates with a
    placeholder explanation. Use this during development to avoid API calls.

    Swap rerank_mock → rerank in main.py when you're ready to go live.
    """
    results = candidates[:5]
    for m in results:
        m["explanation"] = (
            f"'{m['title']}' shares strong thematic and tonal similarities "
            f"with your selections. [{m['genre_string']}]"
        )
    return results
