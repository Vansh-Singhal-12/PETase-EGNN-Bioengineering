"""
process_fireprotdb.py

USAGE:
    python process_fireprotdb.py

INPUT (place in the same folder as this script):
    fireprotdb_raw.csv   -- your full local FireProtDB export

OUTPUT:
    mutations_fireprotdb_verified.csv   -- rows that passed every check,
                                            safe to train on
    fireprotdb_rejected_rows.csv        -- every dropped row + WHY
"""
import re
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# The REAL 6EQE sequence (from the PDB's own SEQRES records), used to verify
# every wild-type letter before it's trusted. Index 0 = residue 1 (Met).
# ---------------------------------------------------------------------------
_SEQRES = """MET ASN PHE PRO ARG ALA SER ARG LEU MET GLN ALA ALA
VAL LEU GLY GLY LEU MET ALA VAL SER ALA ALA ALA THR
ALA GLN THR ASN PRO TYR ALA ARG GLY PRO ASN PRO THR
ALA ALA SER LEU GLU ALA SER ALA GLY PRO PHE THR VAL
ARG SER PHE THR VAL SER ARG PRO SER GLY TYR GLY ALA
GLY THR VAL TYR TYR PRO THR ASN ALA GLY GLY THR VAL
GLY ALA ILE ALA ILE VAL PRO GLY TYR THR ALA ARG GLN
SER SER ILE LYS TRP TRP GLY PRO ARG LEU ALA SER HIS
GLY PHE VAL VAL ILE THR ILE ASP THR ASN SER THR LEU
ASP GLN PRO SER SER ARG SER SER GLN GLN MET ALA ALA
LEU ARG GLN VAL ALA SER LEU ASN GLY THR SER SER SER
PRO ILE TYR GLY LYS VAL ASP THR ALA ARG MET GLY VAL
MET GLY TRP SER MET GLY GLY GLY GLY SER LEU ILE SER
ALA ALA ASN ASN PRO SER LEU LYS ALA ALA ALA PRO GLN
ALA PRO TRP ASP SER SER THR ASN PHE SER SER VAL THR
VAL PRO THR LEU ILE PHE ALA CYS GLU ASN ASP SER ILE
ALA PRO VAL ASN SER SER ALA LEU PRO ILE TYR ASP SER
MET SER ARG ASN ALA LYS GLN PHE LEU GLU ILE ASN GLY
GLY SER HIS SER CYS ALA ASN SER GLY ASN SER ASN GLN
ALA LEU ILE GLY LYS LYS GLY VAL ALA TRP MET LYS ARG
PHE MET ASP ASN ASP THR ARG TYR SER THR PHE ALA CYS
GLU ASN PRO ASN SER THR ARG VAL SER ASP PHE ARG THR
ALA ASN CYS SER LEU GLU HIS HIS HIS HIS HIS HIS""".split()

_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
    'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}
REAL_SEQUENCE = [_THREE_TO_ONE[r] for r in _SEQRES]  # index 0 = residue 1

# Resolved chain in the crystal structure starts at residue 29 (residues
# 1-28 are the unresolved signal peptide). Node 0 in the graph = residue 29.
FIRST_RESOLVED_RESIDUE = 29
LAST_RESOLVED_RESIDUE = 293

# Positions used in benchmark_25.csv -- training rows here would leak into
# the benchmark and must be excluded.
BENCHMARK_PROTECTED_POSITIONS = {
    121, 186, 224, 233, 280,   # FAST-PETase variants
    159, 132, 214, 238,         # DuraPETase / HotPETase additions
    95, 181,                    # Lu et al. calibration benchmarks
    140, 160, 165               
}

SUBSTITUTION_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")

RAW_PATH = "data/fireprotdb_raw.csv"
VERIFIED_OUT = "data/mutations_fireprotdb_verified.csv"
REJECTED_OUT = "data/fireprotdb_rejected_rows.csv"


def parse_substitution(sub):
    """FireProtDB's SUBSTITUTION field is typically like 'S121D' (wt+pos+mut)."""
    if not isinstance(sub, str):
        return None
    m = SUBSTITUTION_RE.match(sub.strip())
    if not m:
        return None
    wt, pos, mt = m.groups()
    return wt.upper(), int(pos), mt.upper()


def real_residue_at(pos):
    if pos < 1 or pos > len(REAL_SEQUENCE):
        return None
    return REAL_SEQUENCE[pos - 1]


def process():
    try:
        df = pd.read_csv(RAW_PATH, low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] '{RAW_PATH}' not found in this folder. Put your FireProtDB "
              f"export there and re-run.")
        sys.exit(1)

    print(f"Loaded {len(df)} raw FireProtDB rows.")

    # Sanity check the columns we depend on actually exist -- FireProtDB's
    # schema is fairly stable but this fails loudly instead of silently if
    # the export has different column names.
    required_cols = {"SUBSTITUTION", "WWPDB", "ORGANISM", "PROTEIN"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] Expected columns missing from your file: {missing}")
        print(f"        Columns actually present: {list(df.columns)}")
        print("        The export's schema may differ -- check column names and "
              "adjust this script's column references accordingly.")
        sys.exit(1)

    verified_rows = []
    rejected_rows = []

    for idx, row in df.iterrows():
        wwpdb = str(row.get("WWPDB", "")).upper()
        organism = str(row.get("ORGANISM", "")).lower()
        protein = str(row.get("PROTEIN", "")).lower()

        is_native_6eqe = "6EQE" in wwpdb
        is_related = ("ideonella" in organism) or ("petase" in protein) or ("cutinase" in protein)

        if not (is_native_6eqe or is_related):
            continue  # irrelevant to this project, skip silently

        parsed = parse_substitution(row.get("SUBSTITUTION"))
        if parsed is None:
            rejected_rows.append({**row.to_dict(), "reject_reason": "could not parse SUBSTITUTION field"})
            continue

        wt, pos, mt = parsed

        if not is_native_6eqe:
            # Real data, but from a homologous protein, not 6EQE itself --
            # can't be checked against the sequence or used with the node
            # mapping without that protein's own structure file. Flagged for
            # a future session, not silently discarded.
            rejected_rows.append({**row.to_dict(),
                                   "reject_reason": f"homologous protein ({protein or organism}), "
                                                     f"not native 6EQE -- needs its own structure to use"})
            continue

        real_aa = real_residue_at(pos)
        if real_aa is None:
            rejected_rows.append({**row.to_dict(), "reject_reason": f"position {pos} outside sequence range"})
            continue
        if real_aa != wt:
            rejected_rows.append({**row.to_dict(),
                                   "reject_reason": f"wild_type mismatch: FireProtDB says {wt}, "
                                                     f"real 6EQE sequence has {real_aa} at position {pos}"})
            continue
        if pos < FIRST_RESOLVED_RESIDUE or pos > LAST_RESOLVED_RESIDUE:
            rejected_rows.append({**row.to_dict(),
                                   "reject_reason": f"position {pos} not resolved in the 6EQE crystal structure "
                                                     f"(resolved range is {FIRST_RESOLVED_RESIDUE}-{LAST_RESOLVED_RESIDUE})"})
            continue
        if pos in BENCHMARK_PROTECTED_POSITIONS:
            rejected_rows.append({**row.to_dict(), "reject_reason": f"position {pos} overlaps benchmark_25.csv -- would leak"})
            continue

        dtm = row.get("DTM")
        ddg = row.get("DDG")
        score, score_source = None, None
        if pd.notna(dtm):
            score, score_source = float(dtm), "DTM"
        elif pd.notna(ddg):
            score, score_source = -float(ddg), "-DDG"  # FireProtDB: +DDG = destabilizing, flipped to match the convention
        else:
            rejected_rows.append({**row.to_dict(), "reject_reason": "no numeric DTM or DDG value"})
            continue

        verified_rows.append({
            "wild_type": wt, "mutation_type": mt, "position_idx": pos,
            "stability_score": score, "score_source": score_source,
            "publication_doi": row.get("PUBLICATION_DOI"),
            "publication_pmid": row.get("PUBLICATION_PMID"),
            "publication_year": row.get("PUBLICATION_YEAR"),
            "fireprotdb_experiment_id": row.get("EXPERIMENT_ID"),
        })

    verified_df = pd.DataFrame(verified_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    verified_df.to_csv(VERIFIED_OUT, index=False)
    rejected_df.to_csv(REJECTED_OUT, index=False)

    print(f"\n[RESULT] {len(verified_df)} rows PASSED all checks -> {VERIFIED_OUT}")
    print(f"[AUDIT]  {len(rejected_df)} rows rejected, with reasons -> {REJECTED_OUT}")
    if len(rejected_df) > 0:
        print("\nRejection reason breakdown (top-level categories):")
        top_level = rejected_df["reject_reason"].str.split(":").str[0]
        print(top_level.value_counts().to_string())

    if len(verified_df) == 0:
        print("\n[WARNING] Zero rows survived. Most likely cause: FireProtDB simply has")
        print("          no entries actually solved on PDB 6EQE specifically (most of its")
        print("          entries are on OTHER proteins). Check fireprotdb_rejected_rows.csv --")
        print("          if most rejections say 'homologous protein', that confirms it, and")
        print("          the real next step is sourcing a structure file for whichever related")
        print("          protein FireProtDB actually has good 6EQE-adjacent coverage for.")


if __name__ == "__main__":
    process()