import os
import csv
import argparse
from Bio.PDB import PDBParser

from src.protein_registry import build_registry
from src.dataset import THREE_TO_ONE, three_to_one

ALL_AA = list("ACDEFGHIKLMNPQRSTVWY")

# Catalytic triad positions (real residue numbers, not node indices) --
# excluded by default since mutating the nucleophile/acid/base directly
# would most likely abolish catalytic function regardless of stability
# effect, which defeats the purpose.
DEFAULT_EXCLUDE_POSITIONS = {160, 206, 237}


def generate_single_point_library(protein_key="6EQE", exclude_positions=None,
                                    output_path="data/screening_round1_singlepoint.csv"):
    if exclude_positions is None:
        exclude_positions = DEFAULT_EXCLUDE_POSITIONS

    registry = build_registry(verbose=False)
    cfg = registry[protein_key]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_key, cfg["pdb_path"])

    first_res = cfg["first_resolved_residue"]
    target_chain = cfg["chain_id"]

    res_names, res_numbers = [], []
    for model in structure:
        for chain in model:
            if chain.id != target_chain:
                continue
            for residue in chain:
                if residue.has_id("CA") and residue.id[0] == ' ':
                    res_names.append(residue.get_resname())
                    res_numbers.append(residue.id[1])  # actual PDB residue number
        break

    print(f"[generate_library] Loaded {len(res_names)} resolved residues for {protein_key} "
          f"(chain {target_chain}, first_res={first_res})")

    rows = []
    skipped_triad = 0
    for resname, res_num in zip(res_names, res_numbers):
        try:
            wt_letter = three_to_one(resname)
        except KeyError:
            continue  # non-standard residue, skip

        if res_num in exclude_positions:
            skipped_triad += 1
            continue

        for mut_letter in ALL_AA:
            if mut_letter == wt_letter:
                continue
            rows.append({
                "wild_type": wt_letter,
                "mutation_type": mut_letter,
                "position_idx": res_num,
                "stability_score": 0.0,  # placeholder -- unused by predict.py, only
                                          # present because PETaseMutationDataset's
                                          # CSV format requires the column to exist.
                "protein_id": protein_key,
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "stability_score", "protein_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[generate_library] Excluded {skipped_triad} triad positions: {sorted(exclude_positions)}")
    print(f"[generate_library] Generated {len(rows)} single-point candidates -> {output_path}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein_key", type=str, default="6EQE")
    ap.add_argument("--output", type=str, default="data/screening_round1_singlepoint.csv")
    ap.add_argument("--exclude_positions", type=str, default="160,206,237",
                     help="Comma-separated residue numbers to exclude (default: catalytic triad)")
    args = ap.parse_args()

    exclude = set(int(x) for x in args.exclude_positions.split(",") if x.strip())
    generate_single_point_library(protein_key=args.protein_key, exclude_positions=exclude,
                                    output_path=args.output)