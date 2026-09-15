"""
Extracts the wild-type amino acid sequence (in residue-number order) from
6EQE.pdb, plus the exact real residue number for each sequence position --
needed because ESM scores by SEQUENCE INDEX (0, 1, 2, ...) while our
project's mutation data uses REAL PDB RESIDUE NUMBERS (e.g., 121, 186),
which are offset from sequence index by first_resolved_residue and are
NOT always perfectly contiguous (gaps can exist in a crystal structure).
Saving both a FASTA file and an explicit index<->position mapping avoids
silently misaligning ESM's output against our mutation data.
"""
import argparse
from Bio.PDB import PDBParser

from src.protein_registry import build_registry
from src.dataset import three_to_one


def extract_sequence(protein_key="6EQE"):
    registry = build_registry(verbose=False)
    cfg = registry[protein_key]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_key, cfg["pdb_path"])

    sequence = []
    residue_numbers = []
    target_chain = cfg["chain_id"]
    for model in structure:
        for chain in model:
            if chain.id != target_chain:
                continue
            for residue in chain:
                if residue.id[0] != ' ':
                    continue
                try:
                    aa = three_to_one(residue.get_resname())
                except KeyError:
                    continue
                sequence.append(aa)
                residue_numbers.append(residue.id[1])
        break

    return "".join(sequence), residue_numbers


def save_outputs(protein_key, fasta_path, mapping_path):
    sequence, residue_numbers = extract_sequence(protein_key)

    with open(fasta_path, "w") as f:
        f.write(f">{protein_key}_mature_domain\n{sequence}\n")

    with open(mapping_path, "w") as f:
        f.write("sequence_index,real_residue_number,wt_amino_acid\n")
        for idx, (pos, aa) in enumerate(zip(residue_numbers, sequence)):
            f.write(f"{idx},{pos},{aa}\n")

    # Check for gaps -- worth knowing if the numbering isn't perfectly
    # contiguous, since that would matter for anyone reading the mapping
    gaps = []
    for i in range(1, len(residue_numbers)):
        if residue_numbers[i] != residue_numbers[i-1] + 1:
            gaps.append((residue_numbers[i-1], residue_numbers[i]))

    print(f"[extract_wt_sequence] Extracted {len(sequence)} residues from {protein_key}")
    print(f"[extract_wt_sequence] Residue number range: {residue_numbers[0]} to {residue_numbers[-1]}")
    print(f"[extract_wt_sequence] Sequence -> {fasta_path}")
    print(f"[extract_wt_sequence] Index<->position mapping -> {mapping_path}")
    if gaps:
        print(f"[extract_wt_sequence] WARNING: {len(gaps)} gap(s) in residue numbering detected: {gaps}")
        print(f"    This is normal for crystal structures (unresolved loop regions) but means")
        print(f"    sequence_index != real_residue_number - first_resolved_residue everywhere.")
        print(f"    ALWAYS use the mapping file, never assume a fixed offset.")
    else:
        print(f"[extract_wt_sequence] No gaps -- numbering is fully contiguous.")

    print(f"\nFirst 60 residues: {sequence[:60]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein_key", type=str, default="6EQE")
    ap.add_argument("--fasta_output", type=str, default="data/6EQE_wt_sequence.fasta")
    ap.add_argument("--mapping_output", type=str, default="data/6EQE_index_mapping.csv")
    args = ap.parse_args()

    save_outputs(args.protein_key, args.fasta_output, args.mapping_output)