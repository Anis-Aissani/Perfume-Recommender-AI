"""
data.py — Load and clean the Fragrantica dataset

This is Step 1 of any data science project:
get raw data into a clean, structured format.
"""

import pandas as pd

CSV_PATH = "fra_cleaned.csv"


def load_perfumes():
    """
    Load the Fragrantica CSV and return a clean DataFrame.

    What we do here:
    - Read the file (it uses semicolons, not commas)
    - Fix European decimal notation  →  3,97 becomes 3.97
    - Filter to perfumes with enough reviews to be trustworthy
    - Parse the notes strings into Python lists
    - Drop anything with no notes at all
    """

    #1 read the CSV
    df = pd.read_csv(CSV_PATH, sep=";", encoding="latin-1")

    #2 rename columns to easier names to use
    df = df.rename(columns={
        "Perfume":      "name",
        "Brand":        "brand",
        "Gender":       "gender",
        "Rating Value": "rating_raw",
        "Rating Count": "rating_count",
        "Year":         "year",
        "Top":          "top",
        "Middle":       "middle",
        "Base":         "base",
    })

    #3 turn , into . for ratings, and confirm that rating counts is numeric
    df["rating"] = df["rating_raw"].astype(str).str.replace(",", ".").astype(float)
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce").fillna(0).astype(int)

    #4 keep perfumes with +100 ratings (to avoid fake/troll perfumes)
    df = df[df["rating_count"] >= 100].copy()

    #parsing notes (top, middle, base) from strings into lists of notes
    def to_list(text):
        if pd.isna(text) or str(text).strip().lower() in ("", "unknown"):
            return []
        return [n.strip().lower() for n in str(text).split(",") if n.strip()]

    df["top_notes"]  = df["top"].apply(to_list)
    df["mid_notes"]  = df["middle"].apply(to_list)
    df["base_notes"] = df["base"].apply(to_list)

    #removing duplicates, and combining all notes into a single list for each perfume
    df["all_notes"] = df.apply(
        lambda row: list(dict.fromkeys(row["top_notes"] + row["mid_notes"] + row["base_notes"])),
        axis=1
    )

    #6 cleaning
    df["name"]  = df["name"].astype(str).str.replace("-", " ").str.strip()
    df["brand"] = df["brand"].astype(str).str.strip().str.title()
    df["url"]   = df["url"].astype(str).str.strip()

    #7 drop irrelevant columns
    df = df[df["all_notes"].map(len) > 0].copy()

    return df.reset_index(drop=True)


def get_brands():
    """Return sorted list of all brands in the filtered dataset."""
    df = load_perfumes()
    return sorted(df["brand"].unique().tolist())
