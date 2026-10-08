import sys, argparse
import numpy as np
from Bio.PDB import PDBParser

ap = argparse.ArgumentParser()
ap.add_argument("receptor_pdb", help="the hydrogen-added receptor PDB, e.g. WT_H.pdb")
ap.add_argument("docked_pdbqt", help="Vina output file, e.g. WT_MHET_out.pdbqt")
ap.add_argument("--ser", type=int, default=132, help="file residue number of Ser160")
ap.add_argument("--tyr", type=int, default=59, help="file residue number of Tyr87")
ap.add_argument("--met", type=int, default=133, help="file residue number of Met161")
args = ap.parse_args()

# ---- receptor atoms of interest ----
chain = next(PDBParser(QUIET=True).get_structure("r", args.receptor_pdb)[0].get_chains())
ser_og = chain[args.ser]["OG"].coord
tyr_n = chain[args.tyr]["N"].coord
met_n = chain[args.met]["N"].coord

# ---- read the docked poses ----
poses, cur = [], None
for line in open(args.docked_pdbqt):
    if line.startswith("MODEL"):
        cur = {"affinity": None, "atoms": []}
    elif line.startswith("REMARK VINA RESULT") and cur is not None:
        cur["affinity"] = float(line.split()[3])
    elif line.startswith(("ATOM", "HETATM")) and cur is not None:
        atype = line[77:].strip()
        if atype.startswith("H"):
            continue                      # skip hydrogens
        el = "O" if atype.startswith("O") else "N" if atype.startswith("N") else "C"
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        cur["atoms"].append((el, xyz))
    elif line.startswith("ENDMDL") and cur is not None:
        poses.append(cur)
        cur = None

def find_ester(atoms):
    """Return (ester carbonyl C, carbonyl O) coordinates, found from bonding geometry."""
    n = len(atoms)
    nbrs = {i: [j for j in range(n) if j != i and
                np.linalg.norm(atoms[i][1] - atoms[j][1]) < 1.75] for i in range(n)}
    for i, (el, xyz) in enumerate(atoms):
        if el != "C" or len(nbrs[i]) != 3:
            continue
        o = [j for j in nbrs[i] if atoms[j][0] == "O"]
        c = [j for j in nbrs[i] if atoms[j][0] == "C"]
        if len(o) == 2 and len(c) == 1:
            alkoxy = [j for j in o if len(nbrs[j]) > 1]
            carbonyl = [j for j in o if len(nbrs[j]) == 1]
            if len(alkoxy) == 1 and len(carbonyl) == 1:
                return xyz, atoms[carbonyl[0]][1]
    return None, None

d = lambda a, b: float(np.linalg.norm(a - b))
print(f"{'pose':>4} {'affinity':>9} {'Ser-OG..ester C':>16} {'C=O..Tyr87 N':>13} {'C=O..Met161 N':>14} {'closest atom to OG':>19}  verdict")
for k, p in enumerate(poses, 1):
    ec, eo = find_ester(p["atoms"])
    closest = min(d(xyz, ser_og) for _, xyz in p["atoms"])
    if ec is None:
        print(f"{k:>4} {p['affinity']:>9.2f}   (ester carbon not found)  closest atom {closest:.1f}")
        continue
    dc, dy, dm = d(ec, ser_og), d(eo, tyr_n), d(eo, met_n)
    ok = dc <= 4.0 and min(dy, dm) <= 3.5
    print(f"{k:>4} {p['affinity']:>9.2f} {dc:>16.2f} {dy:>13.2f} {dm:>14.2f} {closest:>19.2f}  {'PLAUSIBLE' if ok else '-'}")