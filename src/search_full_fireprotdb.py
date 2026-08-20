"""
search_full_fireprotdb.py

Searches the ENTIRE 5.4M-row file (not a sample) for IsPETase entries,
using exact identifiers pulled straight from the 6EQE PDB header -- more
reliable than fuzzy text matching on protein/organism names.

USAGE (from repo root):
    python -m src.search_full_fireprotdb
"""
import pandas as pd

RAW_PATH = "data/fireprotdb_raw.csv"

# Exact identifiers for IsPETase, taken directly from 6EQE's own PDB header:
#   COMPND ... EC: 3.1.1.101
#   DBREF  6EQE A  1  290  UNP  A0A0K8P6T7
TARGET_EC_NUMBER = "3.1.1.101"
TARGET_UNIPROT = "A0A0K8P6T7"

def search():
    print("Reading the full file -- this may take a minute or two on 5.4M rows...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Loaded all {len(df)} rows.\n")

    # 1) Exact EC number match (most reliable -- catches PETase even if
    #    named/labeled inconsistently across different source datasets)
    ec_matches = df[df["EC_NUMBER"].astype(str).str.strip() == TARGET_EC_NUMBER]
    print(f"[EC_NUMBER == '{TARGET_EC_NUMBER}']: {len(ec_matches)} rows")

    # 2) Exact UniProt match
    uniprot_matches = df[df["UNIPROTKB"].astype(str).str.strip() == TARGET_UNIPROT]
    print(f"[UNIPROTKB == '{TARGET_UNIPROT}']: {len(uniprot_matches)} rows")

    # 3) Broad text search across the WHOLE file (not just the 200k sample
    #    from yesterday) as a cross-check/backup
    text_mask = pd.Series(False, index=df.index)
    for col in ["PROTEIN", "ORGANISM", "WWPDB"]:
        text_mask |= df[col].astype(str).str.contains(
            "ideonella|petase|6eqe", case=False, na=False, regex=True
        )
    text_matches = df[text_mask]
    print(f"[Broad text search, full file]: {len(text_matches)} rows")

    # 4) Also check for the WIDER cutinase/esterase family via EC prefix --
    #    even if IsPETase itself isn't in here, related cutinases might be,
    #    which would be a real option for a future homologous-protein path.
    ec_family_mask = df["EC_NUMBER"].astype(str).str.startswith("3.1.1", na=False)
    ec_family_matches = df[ec_family_mask]
    print(f"[EC family 3.1.1.* (broader esterase/cutinase family)]: {len(ec_family_matches)} rows")
    if len(ec_family_matches) > 0:
        print("  Distinct proteins found in this EC family:")
        print(f"  {ec_family_matches['PROTEIN'].dropna().unique()[:30]}")

    combined = pd.concat([ec_matches, uniprot_matches, text_matches]).drop_duplicates(subset=["EXPERIMENT_ID"])
    print(f"\n[COMBINED, deduplicated, all IsPETase-specific hits]: {len(combined)} rows")

    if len(combined) > 0:
        combined.to_csv("data/fireprotdb_ispetase_hits.csv", index=False)
        print("Saved to data/fireprotdb_ispetase_hits.csv -- send this back, this is what we process next.")
        with pd.option_context('display.max_columns', None, 'display.width', 200):
            print(combined[["SUBSTITUTION", "PROTEIN", "ORGANISM", "WWPDB", "DTM", "DDG", "PUBLICATION_DOI"]].to_string())
    else:
        print("\n[CONCLUSION] Zero IsPETase-specific rows anywhere in the full 5.4M-row file,")
        print("             checked by EC number, UniProt ID, AND text search. This is a")
        print("             definitive answer, not a maybe: this FireProtDB dump does not")
        print("             contain IsPETase/6EQE data at all. Next step is the homologous-")
        print("             family or manual-literature path, not more searching here.")

if __name__ == "__main__":
    search()