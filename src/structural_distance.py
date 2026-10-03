import numpy as np
from Bio.PDB import PDBParser
from src.protein_registry import build_registry


def build_distance_matrix(protein_key="6EQE"):
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
    pos_to_idx = {pos: i for i, pos in enumerate(res_numbers)}

    return dist_matrix, pos_to_idx


def classify_pair(pos_a, pos_b, dist_matrix, pos_to_idx,
                   proximal_cutoff=8.0, distal_cutoff=18.0):
    """Returns ('proximal'|'intermediate'|'distal', real_distance_angstroms)."""
    if pos_a not in pos_to_idx or pos_b not in pos_to_idx:
        return "unknown", None
    d = dist_matrix[pos_to_idx[pos_a], pos_to_idx[pos_b]]
    if d <= proximal_cutoff:
        return "proximal", float(d)
    elif d >= distal_cutoff:
        return "distal", float(d)
    return "intermediate", float(d)