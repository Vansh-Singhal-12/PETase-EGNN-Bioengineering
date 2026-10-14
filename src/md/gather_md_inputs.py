r"""
gather_md_inputs.py - collect the final MD input structures and build the Colab run queue.

For every candidate in the shortlist file (rule picks, manual picks, negative controls) it copies the right
structure: NAME_SS.pdb (disulfide rebuilt) if it exists, otherwise NAME_H.pdb. It checks every engineered Cys
for a proper bond, then writes queue.json (one 30 ns run per candidate, seed 1, groups interleaved so partial
results stay balanced). A second FAST-PETase replicate is added as the last job (optional extra control).
WT x3 and FAST-PETase seed 1 are already done and are not queued.

Usage (inside the activated 'dock' environment, from the repo folder):
  python src\md\gather_md_inputs.py --shortlist results\md_shortlist_final.csv --runs-dir C:\docking\runs --out-dir C:\PETase_md_data\inputs
"""
import argparse, csv, json, os, shutil
import numpy as np
from Bio.PDB import PDBParser

ap = argparse.ArgumentParser()
ap.add_argument("--shortlist", required=True)
ap.add_argument("--runs-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--ns", type=float, default=30)
ap.add_argument("--offset", type=int, default=28, help="real position = file residue + offset")
ap.add_argument("--no-extra-fast", action="store_true", help="do not add the second FAST-PETase replicate")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
NATIVE_CYS = {203, 239, 273, 289}

def cys_report(path):
    """Return a list of problems with engineered Cys pairs (empty list = fine)."""
    chain = next(PDBParser(QUIET=True).get_structure("s", path)[0].get_chains())
    cys = [r for r in chain if r.get_resname() in ("CYS", "CYX", "CYM") and "SG" in r]
    problems = []
    for r in cys:
        real = r.id[1] + args.offset
        if real in NATIVE_CYS:
            continue
        others = [o for o in cys if o is not r]
        if not others:
            problems.append(f"Cys{real} has no partner"); continue
        o = min(others, key=lambda x: np.linalg.norm(x["SG"].coord - r["SG"].coord))
        d = float(np.linalg.norm(o["SG"].coord - r["SG"].coord))
        if d >= 2.3 or "HG" in r or "HG" in o:
            problems.append(f"Cys{real}-Cys{o.id[1] + args.offset} NOT bonded (S-S {d:.2f} A)")
    return sorted(set(problems))

rows = [r for r in csv.DictReader(open(args.shortlist, newline="")) if r["role"] != "control"]
groups = {"rule_pick": [], "manual_pick": [], "negative_control": []}
seen = set()
for r in rows:
    if r["candidate_id"] in seen: continue
    seen.add(r["candidate_id"]); groups.setdefault(r["role"], []).append(r)

jobs, warnings = [], []
print(f"{'candidate':<15}{'role':<18}{'input file':<12}disulfide check")
for g in groups.values():
    for r in g:
        cid = r["candidate_id"]
        ss = os.path.join(args.runs_dir, cid, f"{cid}_SS.pdb")
        hh = os.path.join(args.runs_dir, cid, f"{cid}_H.pdb")
        src, tag = (ss, "_SS") if os.path.exists(ss) else (hh, "_H")
        if not os.path.exists(src):
            warnings.append(f"{cid}: no prepared structure found"); print(f"{cid:<15}{r['role']:<18}MISSING"); continue
        dst = f"{cid}_MDinput.pdb"
        shutil.copy(src, os.path.join(args.out_dir, dst))
        prob = cys_report(src)
        if prob: warnings.append(f"{cid}: " + "; ".join(prob))
        print(f"{cid:<15}{r['role']:<18}{tag:<12}{'OK' if not prob else '; '.join(prob)}")
        r["_dst"] = dst

# interleave the groups so that any partial set of finished runs is balanced
order, lists = [], [[r for r in g if "_dst" in r] for g in groups.values()]
while any(lists):
    for l in lists:
        if l: order.append(l.pop(0))
for r in order:
    jobs.append({"name": f"{r['candidate_id']}_s1", "pdb": r["_dst"], "seed": 1, "ns": args.ns, "role": r["role"]})

if not args.no_extra_fast:
    fh = os.path.join(args.runs_dir, "control_FASTPETase", "control_FASTPETase_H.pdb")
    if os.path.exists(fh):
        shutil.copy(fh, os.path.join(args.out_dir, "control_FASTPETase_MDinput.pdb"))
        jobs.append({"name": "FASTPETase_rep2", "pdb": "control_FASTPETase_MDinput.pdb", "seed": 2, "ns": args.ns, "role": "control (extra replicate)"})

with open(os.path.join(args.out_dir, "queue.json"), "w") as f:
    json.dump(jobs, f, indent=1)
print(f"\n{len(jobs)} jobs written to {os.path.join(args.out_dir, 'queue.json')}; first five: " + ", ".join(j["name"] for j in jobs[:5]))
if warnings:
    print("\nWARNINGS - read these before running anything:")
    for w in warnings: print("  ", w)
else:
    print("No warnings: every structure was found and every engineered Cys pair is bonded.")
