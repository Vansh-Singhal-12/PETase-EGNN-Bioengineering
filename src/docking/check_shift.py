r"""
check_shift.py - which atoms moved during the minimisation, and are any near the active site?

Usage:
  python check_shift.py ORIGINAL.pdb MINIMISED_H.pdb
e.g.
  python check_shift.py C:\docking\colabfold_full\candidate_001_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb C:\docking\runs\candidate_001\candidate_001_H.pdb
"""
import argparse
import numpy as np
from Bio.PDB import PDBParser

ap = argparse.ArgumentParser()
ap.add_argument("original")
ap.add_argument("minimised")
ap.add_argument("--ser", type=int, default=132, help="file residue number of Ser160")
ap.add_argument("--top", type=int, default=10)
args = ap.parse_args()

p = PDBParser(QUIET=True)
a = {(r.id[1], at.name): (r.get_resname(), at.coord) for r in next(p.get_structure("a", args.original)[0].get_chains()) for at in r}
b = {(r.id[1], at.name): at.coord for r in next(p.get_structure("b", args.minimised)[0].get_chains()) for at in r}
og = a[(args.ser, "OG")][1]

rows = []
for key, (resname, xyz) in a.items():
    if key in b and not key[1].startswith("H"):
        rows.append((float(np.linalg.norm(b[key] - xyz)), key[0], resname, key[1], float(np.linalg.norm(xyz - og))))
rows.sort(reverse=True)

print(f"{'moved(A)':>9} {'file res':>9} {'real pos':>9} {'residue':>8} {'atom':>6} {'dist to Ser OG (A)':>19}")
for d, num, name, atom, dog in rows[: args.top]:
    print(f"{d:>9.2f} {num:>9} {num + 28:>9} {name:>8} {atom:>6} {dog:>19.1f}")

near = [r for r in rows if r[4] <= 12.0]
print(f"\nAtoms within 12 A of Ser OG (the docking box): {len(near)}")
print(f"  of those, moved more than 1.0 A: {sum(r[0] > 1.0 for r in near)}; more than 2.0 A: {sum(r[0] > 2.0 for r in near)}")
print(f"  largest movement inside the box: {max(r[0] for r in near):.2f} A")
