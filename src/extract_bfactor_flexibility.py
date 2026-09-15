"""
Extracts experimental B-factors from the 6EQE PDB file and maps them to
residue positions -- a free, already-available flexibility signal. 
High B-factor = more flexible/mobile in the crystal structure;
low B-factor = more rigid.

Purpose: PETase thermostabilizing mutations are documented to work partly
by rigidifying flexible loops (e.g., Loop 10, per the D186 mechanism
paper). This flags which of our candidate positions sit in naturally
flexible regions -- a third, independent signal alongside FoldX and the
upcoming pLM score, rather than trusting FoldX's structure-only energy
calculation alone.
"""
import argparse
import csv
import numpy as np
from Bio.PDB import PDBParser

from src.protein_registry import build_registry


def extract_bfactors(protein_key="6EQE"):
    """Returns dict: real_residue_number -> average B-factor across that
    residue's atoms (using all atoms, not just CA, for a fuller picture of
    local mobility -- side chains often show more flexibility signal than
    the backbone CA alone)."""
    registry = build_registry(verbose=False)
    cfg = registry[protein_key]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_key, cfg["pdb_path"])

    bfactors_by_pos = {}
    target_chain = cfg["chain_id"]
    for model in structure:
        for chain in model:
            if chain.id != target_chain:
                continue
            for residue in chain:
                if residue.id[0] != ' ':
                    continue  # skip heteroatoms/waters
                atom_bfactors = [atom.get_bfactor() for atom in residue]
                if atom_bfactors:
                    bfactors_by_pos[residue.id[1]] = float(np.mean(atom_bfactors))
        break

    return bfactors_by_pos


def annotate_candidates(candidates_csv_path, output_path, protein_key="6EQE",
                         flexible_percentile=75):
    bfactors_by_pos = extract_bfactors(protein_key)
    all_bfactors = np.array(list(bfactors_by_pos.values()))

    flex_threshold = float(np.percentile(all_bfactors, flexible_percentile))
    print(f"[extract_bfactor] Whole-protein B-factor stats: "
          f"mean={all_bfactors.mean():.2f}, median={np.median(all_bfactors):.2f}, "
          f"{flexible_percentile}th pct={flex_threshold:.2f}")
    print(f"[extract_bfactor] Positions with B-factor >= {flex_threshold:.2f} flagged as 'flexible region'\n")

    with open(candidates_csv_path) as f:
        rows = list(csv.DictReader(f))

    annotated = []
    n_flexible = 0
    for r in rows:
        pos = int(r["position_idx"].split(",")[0]) if "position_idx" in r else int(r["position_idx"])
        bfactor = bfactors_by_pos.get(pos)
        is_flexible = bfactor is not None and bfactor >= flex_threshold
        if is_flexible:
            n_flexible += 1

        new_row = dict(r)
        new_row["bfactor"] = bfactor if bfactor is not None else ""
        new_row["is_flexible_region"] = is_flexible
        annotated.append(new_row)

    fieldnames = list(annotated[0].keys()) if annotated else []
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated)

    print(f"[extract_bfactor] Annotated {len(annotated)} candidates -> {output_path}")
    print(f"[extract_bfactor] {n_flexible}/{len(annotated)} ({100*n_flexible/len(annotated):.1f}%) "
          f"are in flexible regions (B-factor >= {flexible_percentile}th percentile)")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, default="results/foldx_singlepoint_ranked.csv")
    ap.add_argument("--output", type=str, default="results/foldx_singlepoint_ranked_bfactor.csv")
    ap.add_argument("--protein_key", type=str, default="6EQE")
    ap.add_argument("--flexible_percentile", type=int, default=75)
    args = ap.parse_args()

    annotate_candidates(args.candidates, args.output, protein_key=args.protein_key,
                         flexible_percentile=args.flexible_percentile)