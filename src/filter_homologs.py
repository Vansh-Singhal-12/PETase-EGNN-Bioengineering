"""
filter_homologs.py

Pulls out everything FireProtDB has on cutinase-family homologs of IsPETase
(Cutinase 1, Cutinase cut2, LCC, etc.) -- separate from the IsPETase-specific
search we already ran. This tells us exactly which protein(s) and PDB
structure(s) are worth pursuing.

USAGE (from repo root):
    python -m src.filter_homologs
"""
import pandas as pd

RAW_PATH = "data/fireprotdb_raw.csv"
OUT_PATH = "data/fireprotdb_homolog_hits.csv"

def filter_homologs():
    print("Reading full file...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Loaded {len(df)} rows.\n")

    mask = df["PROTEIN"].astype(str).str.contains(
        "cutinase|LCC|leaf-branch|leaf branch|compost", case=False, na=False, regex=True
    )
    hits = df[mask]
    print(f"[Cutinase-family matches]: {len(hits)} rows")

    if len(hits) == 0:
        print("No cutinase-family entries found. Widening to full EC 3.1.1.* family for reference:")
        ec_mask = df["EC_NUMBER"].astype(str).str.startswith("3.1.1", na=False)
        print(df[ec_mask]["PROTEIN"].dropna().unique())
        return

    print("\nDistinct proteins found:")
    print(hits["PROTEIN"].dropna().unique())
    print("\nDistinct WWPDB (structure) IDs found for these proteins:")
    print(hits["WWPDB"].dropna().unique())
    print("\nDistinct UNIPROTKB IDs found:")
    print(hits["UNIPROTKB"].dropna().unique())

    hits.to_csv(OUT_PATH, index=False)
    print(f"\nSaved all matching rows to {OUT_PATH} -- send this back.")

    with pd.option_context('display.max_columns', None, 'display.width', 250):
        cols = ["SUBSTITUTION", "PROTEIN", "ORGANISM", "WWPDB", "UNIPROTKB", "DTM", "DDG", "TM", "PUBLICATION_DOI"]
        print("\nFull preview:")
        print(hits[cols].to_string())

if __name__ == "__main__":
    filter_homologs()