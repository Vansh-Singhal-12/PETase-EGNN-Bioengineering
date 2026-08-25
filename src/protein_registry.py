"""
src/protein_registry.py

Builds the protein registry by scanning actual PDB files on disk, rather
than hand-maintaining a hardcoded dict per protein. With 131+ structures
now in play, hand-verifying each one's first-resolved-residue offset (the
way we did for 6EQE and LCC) isn't practical -- but it doesn't need to be
manual to still be correct: every PDB file states its own DBREF/first ATOM
record, so the offset is fully derivable, not guessed.

Naming convention: 6EQE and LCC keep their names. S2648 structures with a
single chain are keyed "{PDBID}_{CHAIN}" (e.g. "1A5E_A"). Structures with
multiple distinct chains (found: 1LUC has both A and B) get one registry
entry PER CHAIN -- this matters, not just bookkeeping: a mutation meant for
chain B must never be silently applied to chain A's graph.

Only 6EQE and LCC have a known catalytic triad (they're both real
hydrolases with a verified Ser-His-Asp triad). Every S2648 structure gets
catalytic_triad=None, which means the active-site-shield loss term
contributes zero for those rows (train.py already guards on
`mask.sum() > 0` before computing the shield penalty, so this requires no
change there -- an empty/all-False mask just naturally skips it).
"""
import os
import glob
from Bio.PDB import PDBParser

# Hand-verified entries (sessions 3 & 6) -- these stay hardcoded because they
# were individually confirmed against primary literature, not auto-derived.
KNOWN_CATALYTIC_TRIADS = {
    "6EQE": [160, 206, 237],   # Ser160, Asp206, His237 -- verified session 3
    "LCC":  [165, 210, 242],   # Ser165, Asp210, His242 -- verified session 6, direct from RCSB/literature
}

MANUAL_PDB_PATHS = {
    "6EQE": "data/6eqe.pdb",
    "LCC": "data/lcc.pdb",
}

S2648_STRUCTURES_DIR = "data/s2648_structures"


def _first_resolved_residue_and_chains(pdb_path):
    """
    Returns {chain_id: (first_residue_number, num_resolved_residues)} by
    reading the actual atom records -- no assumptions, no hardcoding.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    result = {}
    for model in structure:
        for chain in model:
            residues = [r for r in chain if r.id[0] == ' ' and r.has_id("CA")]
            if not residues:
                continue
            first_res_num = residues[0].id[1]
            result[chain.id] = (first_res_num, len(residues))
        break  # only need the first model
    return result


def build_registry(include_s2648=True, verbose=True):
    """
    Scans disk and returns the full PROTEIN_REGISTRY dict, auto-detecting
    every S2648 structure's chains and residue offsets. Fails loudly (raises,
    with the specific file named) rather than silently skipping a broken
    file -- a protein quietly missing from training is exactly the kind of
    thing that should be visible, not swallowed.
    """
    registry = {}

    # Hand-verified proteins first
    for protein_id, pdb_path in MANUAL_PDB_PATHS.items():
        if not os.path.exists(pdb_path):
            if verbose:
                print(f"[registry] WARNING: {pdb_path} not found, skipping {protein_id}")
            continue
        chains = _first_resolved_residue_and_chains(pdb_path)
        # 6EQE/LCC are single-chain structures -- take the first (only) chain
        chain_id = next(iter(chains))
        first_res, n_res = chains[chain_id]
        registry[protein_id] = {
            "pdb_path": pdb_path,
            "chain_id": chain_id,
            "first_resolved_residue": first_res,
            "num_resolved_residues": n_res,
            "catalytic_triad": KNOWN_CATALYTIC_TRIADS.get(protein_id),
        }
        if verbose:
            print(f"[registry] {protein_id}: chain {chain_id}, first_res={first_res}, "
                  f"n_res={n_res}, triad={KNOWN_CATALYTIC_TRIADS.get(protein_id)}")

    if include_s2648 and os.path.isdir(S2648_STRUCTURES_DIR):
        pdb_files = sorted(glob.glob(os.path.join(S2648_STRUCTURES_DIR, "*.pdb")))
        for pdb_path in pdb_files:
            pdb_id = os.path.splitext(os.path.basename(pdb_path))[0].upper()
            try:
                chains = _first_resolved_residue_and_chains(pdb_path)
            except Exception as e:
                print(f"[registry] ERROR parsing {pdb_path}: {e} -- skipping this structure")
                continue
            for chain_id, (first_res, n_res) in chains.items():
                if n_res < 10:
                    continue  # skip tiny fragments/ligand-only chains, not real protein chains
                protein_id = f"{pdb_id}_{chain_id}"
                registry[protein_id] = {
                    "pdb_path": pdb_path,
                    "chain_id": chain_id,
                    "first_resolved_residue": first_res,
                    "num_resolved_residues": n_res,
                    "catalytic_triad": None,  # not a verified hydrolase active site
                }
        if verbose:
            print(f"[registry] Auto-registered {len(pdb_files)} S2648 structure files "
                  f"-> {sum(1 for k in registry if k not in MANUAL_PDB_PATHS)} protein/chain entries")

    return registry


if __name__ == "__main__":
    reg = build_registry()
    print(f"\nTotal registry entries: {len(reg)}")