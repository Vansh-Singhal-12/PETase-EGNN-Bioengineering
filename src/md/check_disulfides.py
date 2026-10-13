"""
check_disulfides.py - are the Cys pairs in each prepared structure really joined as disulfide bonds?

A disulfide only exists in the MD if the two sulfur atoms were close enough (< 3 A) when the structure was
prepared. Otherwise the cysteines are treated as free thiols (they carry an HG hydrogen) and the simulation
silently tests the REDUCED pair, not the disulfide.

Usage (inside the activated 'dock' environment):
  python check_disulfides.py --runs-dir C:\docking\runs --ids candidate_057 candidate_058 candidate_062 control_WT
  python check_disulfides.py --runs-dir C:\docking\runs --csv results\md_shortlist_final.csv
  python check_disulfides.py --runs-dir C:\docking\runs --suffix _SS --ids candidate_057 candidate_059
"""
import argparse, csv, os
import numpy as np
from Bio.PDB import PDBParser

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", required=True)
ap.add_argument("--ids", nargs="*", default=[])
ap.add_argument("--csv", default="", help="md_shortlist csv with a candidate_id column")
ap.add_argument("--offset", type=int, default=28, help="real position = file residue + offset")
ap.add_argument("--suffix", default="_H", help="file ending to check: _H (default) or _SS (after make_disulfide.py)")
args = ap.parse_args()

ids = list(args.ids)
if args.csv:
    ids += [r["candidate_id"] for r in csv.DictReader(open(args.csv, newline=""))]
ids = [i for n, i in enumerate(ids) if i not in ids[:n]]

parser = PDBParser(QUIET=True)
for cid in ids:
    path = os.path.join(args.runs_dir, cid, f"{cid}{args.suffix}.pdb")
    if not os.path.exists(path):
        print(f"{cid:<20} (no prepared structure at {path})"); continue
    chain = next(parser.get_structure("s", path)[0].get_chains())
    cys = [r for r in chain if r.get_resname() in ("CYS", "CYX", "CYM") and "SG" in r]
    done, lines = set(), []
    for a in cys:
        others = [b for b in cys if b is not a]
        if not others:
            lines.append(f"  Cys{a.id[1] + args.offset}: free (no other Cys)"); continue
        b = min(others, key=lambda r: np.linalg.norm(r["SG"].coord - a["SG"].coord))
        d = float(np.linalg.norm(b["SG"].coord - a["SG"].coord))
        pair = tuple(sorted((a.id[1], b.id[1])))
        if pair in done: continue
        done.add(pair)
        thiol = ("HG" in a) or ("HG" in b)
        status = "BONDED" if (d < 2.3 and not thiol) else "NOT BONDED (free thiols)" if d < 4.5 or thiol else "NOT BONDED"
        lines.append(f"  Cys{pair[0] + args.offset}-Cys{pair[1] + args.offset}: S-S {d:.2f} A, HG hydrogens {'present' if thiol else 'absent'} -> {status}")
    print(f"{cid:<20} {len(cys)} Cys")
    print("\n".join(lines) if lines else "  (no cysteines)")
