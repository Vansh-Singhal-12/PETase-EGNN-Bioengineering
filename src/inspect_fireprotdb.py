"""
inspect_fireprotdb.py

USAGE (from repo root):
    python -m src.inspect_fireprotdb
"""
import pandas as pd

RAW_PATH = "data/fireprotdb_raw.csv"

def inspect():
    df = pd.read_csv(RAW_PATH, low_memory=False, nrows=200000)  # sample, file is huge
    print(f"Loaded a {len(df)}-row SAMPLE (file is 5.4M rows total, not reading it all yet)\n")

    print("=" * 70)
    print("ALL COLUMN NAMES IN YOUR ACTUAL FILE:")
    print("=" * 70)
    print(list(df.columns))

    print("\n" + "=" * 70)
    print("FIRST 5 FULL ROWS (raw, unfiltered):")
    print("=" * 70)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.head(5).to_string())

    # Try to find whichever column looks like it holds the mutation string
    print("\n" + "=" * 70)
    print("SAMPLE VALUES from likely mutation/substitution columns:")
    print("=" * 70)
    candidate_cols = [c for c in df.columns if any(
        kw in c.upper() for kw in ["SUBST", "MUTAT", "VARIANT", "MUTANT"]
    )]
    for c in candidate_cols:
        print(f"\n  Column '{c}' -- 10 sample values:")
        print(f"  {df[c].dropna().unique()[:10]}")

    # Same for protein/organism identity columns
    print("\n" + "=" * 70)
    print("SAMPLE VALUES from protein/organism/structure identity columns:")
    print("=" * 70)
    id_cols = [c for c in df.columns if any(
        kw in c.upper() for kw in ["PROTEIN", "ORGANISM", "PDB", "UNIPROT", "NAME", "GENE"]
    )]
    for c in id_cols:
        print(f"\n  Column '{c}' -- 10 sample values:")
        print(f"  {df[c].dropna().unique()[:10]}")

    # Specifically hunt for anything PETase/Ideonella/cutinase related, using
    # a broad search across ALL text columns, not just the ones that were guessed.
    print("\n" + "=" * 70)
    print("BROAD SEARCH across ALL columns for 'ideonella'/'petase'/'cutinase'/'6eqe':")
    print("=" * 70)
    found_any = False
    for c in df.columns:
        if df[c].dtype == object:
            mask = df[c].astype(str).str.contains(
                "ideonella|petase|cutinase|6eqe", case=False, na=False, regex=True
            )
            if mask.any():
                found_any = True
                print(f"\n  Column '{c}' has {mask.sum()} matching rows in this sample. Examples:")
                print(df.loc[mask, c].unique()[:10])
    if not found_any:
        print("\n  No matches found in this 200k-row sample at all.")
        print("  This could mean: (a) matches exist later in the file (it's 5.4M rows,")
        print("  It only sampled the first 200k), or (b) this bulk dump genuinely has")
        print("  no PETase-family entries and is dominated by the unrelated megascale")
        print("  dataset. Re-run with a larger sample or full file if needed.")


if __name__ == "__main__":
    inspect()