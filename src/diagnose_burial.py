"""
Diagnostic v2: generalizes the burial-bias check to rows with ANY number of
mutated positions (1 for Round 1 singles, 2 for Round 2 pairwise combos,
3-5 for later higher-order combos) -- same burial proxy as v1 (8A contact
degree), plus a position-frequency summary to make cluster-repetition
visible directly rather than requiring a manual scan.

Works on either round1_ranked.csv (columns: predicted_score) or
round2_combo_ranked.csv (columns: combo_score, epistasis_score) --
auto-detects which score column to use.
"""
import argparse
import csv
import numpy as np
from collections import Counter
from Bio.PDB import PDBParser

from src.protein_registry import build_registry

BULKY_HYDROPHOBIC = set("ILVFWMY")
SMALL_FLEXIBLE = set("GPAS")


def compute_burial_degrees(protein_key="6EQE", cutoff=8.0):
    registry = build_registry(verbose=False)
    cfg = registry[protein_key]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_key, cfg["pdb_path"])

    coords, res_numbers = [], []
    target_chain = cfg["chain_id"]
    for model in structure:
        for chain in model:
            if chain.id != target_chain:
                continue
            for residue in chain:
                if residue.has_id("CA") and residue.id[0] == ' ':
                    coords.append(residue["CA"].get_coord())
                    res_numbers.append(residue.id[1])
        break

    coords = np.array(coords)
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    degrees = ((dist_matrix <= cutoff) & (dist_matrix > 0)).sum(axis=1)

    return dict(zip(res_numbers, degrees)), degrees


def detect_score_column(rows):
    """Round 1 uses 'predicted_score'; Round 2 uses 'combo_score' (and optionally
    'epistasis_score', selectable separately)."""
    if not rows:
        return None
    if "predicted_score" in rows[0]:
        return "predicted_score"
    if "combo_score" in rows[0]:
        return "combo_score"
    raise ValueError("Couldn't find a recognized score column in this CSV.")


def diagnose(ranked_csv_path, top_n=50, protein_key="6EQE", degree_cutoff=8.0,
             sort_by=None):
    burial_by_pos, all_degrees = compute_burial_degrees(protein_key, degree_cutoff)
    p75_degree = float(np.percentile(all_degrees, 75))

    print(f"[diagnose_burial v2] 75th pct degree (buried threshold): {p75_degree:.1f}\n")

    with open(ranked_csv_path) as f:
        rows = list(csv.DictReader(f))

    score_col = sort_by or detect_score_column(rows)
    # Filter out rows with missing/empty score (can happen for epistasis_score
    # on rows where a component single-point score was missing)
    rows = [r for r in rows if r.get(score_col, "") not in ("", None)]
    rows.sort(key=lambda r: float(r[score_col]), reverse=True)
    top_rows = rows[:top_n]

    print(f"[diagnose_burial v2] Analyzing top {len(top_rows)} rows by '{score_col}'\n")
    print(f"{'WT->Mut':<20}{'Positions':<16}{'Score':<10}{'#Buried':<10}{'#BulkySmall'}")
    print("-" * 75)

    position_counter = Counter()
    n_rows_with_any_buried = 0
    n_rows_all_bulky_small = 0
    total_legs = 0
    total_buried_legs = 0
    total_bulky_small_legs = 0

    for r in top_rows:
        wt_parts = r["wild_type"].split(";")
        mut_parts = r["mutation_type"].split(";")
        pos_parts = [int(p) for p in r["position_idx"].split(",")]
        score = float(r[score_col])

        n_buried_legs_this_row = 0
        n_bulky_small_legs_this_row = 0

        for wt, mut, pos in zip(wt_parts, mut_parts, pos_parts):
            position_counter[pos] += 1
            total_legs += 1
            degree = burial_by_pos.get(pos)
            if degree is not None and degree >= p75_degree:
                n_buried_legs_this_row += 1
                total_buried_legs += 1
            if wt in BULKY_HYDROPHOBIC and mut in SMALL_FLEXIBLE:
                n_bulky_small_legs_this_row += 1
                total_bulky_small_legs += 1

        if n_buried_legs_this_row > 0:
            n_rows_with_any_buried += 1
        if n_bulky_small_legs_this_row == len(wt_parts):
            n_rows_all_bulky_small += 1

        wt_mut_str = ";".join(f"{w}->{m}" for w, m in zip(wt_parts, mut_parts))
        pos_str = ",".join(str(p) for p in pos_parts)
        print(f"{wt_mut_str:<20}{pos_str:<16}{score:<10.3f}"
              f"{n_buried_legs_this_row}/{len(pos_parts):<8}{n_bulky_small_legs_this_row}/{len(pos_parts)}")

    print("-" * 75)
    print(f"\n[diagnose_burial v2] SUMMARY (top {len(top_rows)} rows, {total_legs} total mutation-legs):")
    print(f"  Rows with >=1 buried leg          : {n_rows_with_any_buried} ({100*n_rows_with_any_buried/len(top_rows):.0f}%)")
    print(f"  Rows where ALL legs are bulky->small: {n_rows_all_bulky_small} ({100*n_rows_all_bulky_small/len(top_rows):.0f}%)")
    print(f"  Buried legs / total legs            : {total_buried_legs}/{total_legs} ({100*total_buried_legs/max(1,total_legs):.0f}%)")
    print(f"  Bulky->small legs / total legs      : {total_bulky_small_legs}/{total_legs} ({100*total_bulky_small_legs/max(1,total_legs):.0f}%)")

    print(f"\n[diagnose_burial v2] POSITION FREQUENCY (how often each position recurs in the top {len(top_rows)}):")
    for pos, count in position_counter.most_common(15):
        degree = burial_by_pos.get(pos, "?")
        buried_flag = "BURIED" if isinstance(degree, (int, np.integer)) and degree >= p75_degree else ""
        print(f"    Position {pos:<5} appears {count:>3}x  (degree={degree} {buried_flag})")

    n_unique_positions = len(position_counter)
    print(f"\n  Unique positions represented: {n_unique_positions} "
          f"(out of {total_legs} total legs -- lower means more repetition/less diversity)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_csv", type=str, default="results/round1_ranked.csv")
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--protein_key", type=str, default="6EQE")
    ap.add_argument("--sort_by", type=str, default=None,
                     help="Column to sort/rank by (e.g. 'combo_score' or 'epistasis_score'). "
                          "Auto-detected if omitted.")
    args = ap.parse_args()

    diagnose(args.ranked_csv, top_n=args.top_n, protein_key=args.protein_key, sort_by=args.sort_by)