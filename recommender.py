"""
recommender.py — The recommendation engine

Given what a user likes, find perfumes with similar notes.

Technique: Cosine Similarity
  - Represent each perfume as a row of 0s and 1s (one column per note)
  - Do the same for the user's preferences
  - Measure how "close" each perfume is to the user vector
  - Return the closest ones
"""

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

from data import load_perfumes


@lru_cache(maxsize=1)
def build_matrix():
    """
    Load the data and build the feature matrix.
    Cached — runs once on startup, not on every request.

    Steps:
      1. Load the clean DataFrame
      2. MultiLabelBinarizer turns note lists into binary vectors
         ["rose", "musk"]  →  [0, 1, 0, 0, 1, 0, ...]
      3. Return the matrix alongside the fitted binarizer
    """
    df = load_perfumes()
    mlb = MultiLabelBinarizer()
    note_matrix = mlb.fit_transform(df["all_notes"])
    return df, mlb, note_matrix


def recommend(selected_notes, gender=None, brands=None, top_n=8):
    """
    Find the top_n perfumes most similar to the selected notes.

    Optional filters:
      gender — "women", "men", or "unisex"
      brands — list of brand names to restrict results to
    """
    df, mlb, note_matrix = build_matrix()

    # Apply filters before scoring
    mask = np.ones(len(df), dtype=bool)

    if gender:
        mask &= (df["gender"] == gender).values

    if brands:
        mask &= df["brand"].isin(brands).values

    filtered_df     = df[mask].copy()
    filtered_matrix = note_matrix[mask]

    if len(filtered_df) == 0:
        return []

    #encode user preferences
    user_vector = mlb.transform([selected_notes])

    #cosine similarity: 1.0 = identical direction, 0.0 = nothing in common
    scores = cosine_similarity(user_vector, filtered_matrix)[0]

    filtered_df = filtered_df.copy()
    filtered_df["score"] = scores
    top = filtered_df.nlargest(top_n, "score")

    results = []
    for _, row in top.iterrows():
        pct = round(float(row["score"]) * 100, 1)
        results.append({
            "name":          row["name"],
            "brand":         row["brand"],
            "gender":        row["gender"],
            "rating":        round(float(row["rating"]), 2),
            "rating_count":  int(row["rating_count"]),
            "year":          int(row["year"]) if str(row.get("year", "")) not in ("nan", "", "None") else None,
            "top_notes":     row["top_notes"][:5],
            "mid_notes":     row["mid_notes"][:5],
            "base_notes":    row["base_notes"][:4],
            "url":           row["url"],
            "score_pct":     pct,
            "matched_notes": [n for n in selected_notes if n in row["all_notes"]],
        })

    return results


def get_notes_for_ui():
    """Return notes grouped by tier for the chip selectors."""
    from collections import Counter

    df, _, _ = build_matrix()

    top_counter  = Counter(n for notes in df["top_notes"]  for n in notes)
    mid_counter  = Counter(n for notes in df["mid_notes"]  for n in notes)
    base_counter = Counter(n for notes in df["base_notes"] for n in notes)

    top_notes  = [n for n, _ in top_counter.most_common(24)]
    base_notes = [n for n, _ in base_counter.most_common(24) if n not in top_notes]
    mid_notes  = [n for n, _ in mid_counter.most_common(28)
                  if n not in top_notes and n not in base_notes]

    return {"top": top_notes, "mid": mid_notes[:24], "base": base_notes[:20]}


def get_stats():
    """Basic dataset stats shown in the header."""
    df, mlb, note_matrix = build_matrix()
    return {
        "total_perfumes": len(df),
        "total_notes":    len(mlb.classes_),
        "feature_dims":   note_matrix.shape[1],
    }
