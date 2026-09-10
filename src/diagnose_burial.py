"""
Diagnostic: checks whether top-ranked screening candidates are dominated by
a "bulky residue -> small residue" shortcut at BURIED positions specifically
(where that substitution is most likely to be genuinely destabilizing due to
cavity formation / loss of packing, rather than a real structural insight).

Burial proxy: uses the same 8A Euclidean-distance graph already built by
dataset.py's _load_protein_as_graph -- a residue's "degree" (number of other
Calpha atoms within 8A) is a simple, dependency-free stand-in for solvent
burial. Buried core residues typically have high degree (many neighbors
packed around them); surface-exposed residues typically have low degree.
This is a proxy, not a substitute for real solvent-accessible-surface-area
(SASA) calculation -- flagged as such in the output.

No external tools required (no DSSP binary needed), reuses existing PDB
parsing via protein_registry.
"""
import argparse
import csv
import numpy as np
from Bio.PDB import PDBParser

from src.protein_registry import build_registry
from src.dataset import three_to_one

BULKY_HYDROPHOBIC = set("ILVFWMY")
SMALL_FLEXIBLE = set("GPAS")


def compute_burial_degrees(protein_key="6EQE", cutoff=8.0):
    """Returns dict: real_residue_number -> degree (count of neighbors within cutoff)."""
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


def diagnose(ranked_csv_path, top_n=50, protein_key="6EQE", degree_cutoff=8.0):
    burial_by_pos, all_degrees = compute_burial_degrees(protein_key, degree_cutoff)
    median_degree = float(np.median(all_degrees))
    p75_degree = float(np.percentile(all_degrees, 75))

    print(f"[diagnose_burial] Whole-protein degree stats (8A contact count): "
          f"median={median_degree:.1f}, 75th pct={p75_degree:.1f}, "
          f"max={all_degrees.max()}, min={all_degrees.min()}")
    print(f"[diagnose_burial] Using >75th percentile ({p75_degree:.1f} neighbors) as 'likely buried' threshold.\n")

    rows = []
    with open(ranked_csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    top_rows = rows[:top_n]

    print(f"{'WT->Mut':<10}{'Pos':<6}{'Score':<10}{'Degree':<8}{'Buried?':<10}{'Pattern'}")
    print("-" * 70)

    n_bulky_to_small = 0
    n_bulky_to_small_buried = 0
    n_buried_total = 0

    for r in top_rows:
        wt = r["wild_type"]
        mut = r["mutation_type"]
        pos = int(r["position_idx"].split(",")[0])
        score = float(r["predicted_score"])
        degree = burial_by_pos.get(pos, None)

        if degree is None:
            print(f"{wt}->{mut:<7}{pos:<6}{score:<10.3f}{'?':<8}{'unknown (no PDB match)'}")
            continue

        is_buried = degree >= p75_degree
        pattern = ""
        if ";" not in wt and ";" not in mut:
            if wt in BULKY_HYDROPHOBIC and mut in SMALL_FLEXIBLE:
                pattern = "BULKY->SMALL"
                n_bulky_to_small += 1
                if is_buried:
                    n_bulky_to_small_buried += 1

        if is_buried:
            n_buried_total += 1

        print(f"{wt}->{mut:<7}{pos:<6}{score:<10.3f}{degree:<8}{'YES' if is_buried else 'no':<10}{pattern}")

    print("-" * 70)
    print(f"\n[diagnose_burial] SUMMARY (top {len(top_rows)} candidates):")
    print(f"  Bulky-hydrophobic -> small/flexible substitutions : {n_bulky_to_small} "
          f"({100*n_bulky_to_small/len(top_rows):.0f}% of top {len(top_rows)})")
    print(f"  ...of those, at BURIED positions (>75th pct degree): {n_bulky_to_small_buried} "
          f"({100*n_bulky_to_small_buried/max(1,n_bulky_to_small):.0f}% of the bulky->small group)")
    print(f"  Total candidates at buried positions (any mutation type): {n_buried_total} "
          f"({100*n_buried_total/len(top_rows):.0f}% of top {len(top_rows)})")
    print(f"\n  [interpretation] If n_bulky_to_small is a large fraction of the top list AND")
    print(f"  most of those are at buried positions, this supports the hypothesis that the")
    print(f"  model is applying a 'shrink buried residues' shortcut rather than genuine")
    print(f"  position-specific structural reasoning. This is a PROXY (contact-count burial,")
    print(f"  not true SASA) -- treat as a flag for closer FoldX/ColabFold scrutiny on these")
    print(f"  specific candidates, not as a final verdict.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_csv", type=str, default="results/round1_ranked.csv")
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--protein_key", type=str, default="6EQE")
    args = ap.parse_args()

    diagnose(args.ranked_csv, top_n=args.top_n, protein_key=args.protein_key)