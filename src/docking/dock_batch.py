r"""
dock_batch.py - dock MHET into every ColabFold rank_001 structure and summarise the poses.

Usage (inside the activated 'dock' environment, from the folder holding this script,
prep_receptor.py and MHET.pdbqt):
  python dock_batch.py --pdb-dir C:\docking\structures --out-dir C:\docking\runs --ligand MHET.pdbqt

It is safe to stop and restart: finished steps are skipped.
"""
import argparse, csv, glob, os, subprocess, sys, time, traceback
import numpy as np
from Bio.PDB import PDBParser

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--pdb-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--ligand", default=os.path.join(HERE, "MHET.pdbqt"))
ap.add_argument("--vina", default=r"C:\vina\vina.exe")
ap.add_argument("--prep-script", default=os.path.join(HERE, "prep_receptor.py"))
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--exhaustiveness", type=int, default=32)
ap.add_argument("--box-size", type=float, default=22.0)
ap.add_argument("--ser", type=int, default=132, help="file residue number of Ser160")
ap.add_argument("--tyr", type=int, default=59, help="file residue number of Tyr87")
ap.add_argument("--met", type=int, default=133, help="file residue number of Met161")
ap.add_argument("--only", default="", help="only run candidates whose id contains this text")
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
parser = PDBParser(QUIET=True)
dist = lambda a, b: float(np.linalg.norm(a - b))

def run(cmd, logfile=None):
    with open(logfile, "w") if logfile else open(os.devnull, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}"
                           + (f"  (see {logfile})" if logfile else ""))

def find_ester(atoms):
    n = len(atoms)
    nbrs = {i: [j for j in range(n) if j != i and dist(atoms[i][1], atoms[j][1]) < 1.75] for i in range(n)}
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

def analyze(receptor_h_pdb, docked_pdbqt):
    chain = next(parser.get_structure("r", receptor_h_pdb)[0].get_chains())
    ser_og, tyr_n, met_n = chain[args.ser]["OG"].coord, chain[args.tyr]["N"].coord, chain[args.met]["N"].coord
    poses, cur = [], None
    for line in open(docked_pdbqt):
        if line.startswith("MODEL"):
            cur = {"affinity": None, "atoms": []}
        elif line.startswith("REMARK VINA RESULT") and cur is not None:
            cur["affinity"] = float(line.split()[3])
        elif line.startswith(("ATOM", "HETATM")) and cur is not None:
            t = line[77:].strip()
            if t.startswith("H"):
                continue
            el = "O" if t.startswith("O") else "N" if t.startswith("N") else "C"
            cur["atoms"].append((el, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
        elif line.startswith("ENDMDL") and cur is not None:
            poses.append(cur); cur = None
    out = []
    for p in poses:
        ec, eo = find_ester(p["atoms"])
        if ec is None:
            out.append({"aff": p["affinity"], "dC": None, "plaus": False}); continue
        dc, dy, dm = dist(ec, ser_og), dist(eo, tyr_n), dist(eo, met_n)
        out.append({"aff": p["affinity"], "dC": dc, "plaus": dc <= 4.0 and min(dy, dm) <= 3.5})
    return out

def summarise(cid, per_seed):
    allp = [p for s in per_seed.values() for p in s]
    best = [min(p["aff"] for p in s) for s in per_seed.values()]
    plaus = [p for p in allp if p["plaus"]]
    dcs = [p["dC"] for p in allp if p["dC"] is not None]
    return {
        "candidate_id": cid,
        "best_affinity": round(min(best), 2),
        "mean_best_affinity": round(float(np.mean(best)), 2),
        "plausible_poses": f"{len(plaus)}/{len(allp)}",
        "seeds_with_plausible": f"{sum(any(p['plaus'] for p in s) for s in per_seed.values())}/{len(per_seed)}",
        "best_affinity_plausible": round(min(p["aff"] for p in plaus), 2) if plaus else "",
        "min_SerOG_esterC": round(min(dcs), 2) if dcs else "",
    }

pdbs = sorted(glob.glob(os.path.join(args.pdb_dir, "*_unrelaxed_rank_001_*.pdb")))
pdbs = [p for p in pdbs if args.only in os.path.basename(p)]
print(f"{len(pdbs)} structures to process; seeds {args.seeds}, exhaustiveness {args.exhaustiveness}")
rows, errors = [], []
consecutive_fail = 0
for n, pdb in enumerate(pdbs, 1):
    cid = os.path.basename(pdb).split("_unrelaxed_rank_001")[0]
    wd = os.path.join(args.out_dir, cid); os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    try:
        hpdb = os.path.join(wd, f"{cid}_H.pdb")
        prefix = os.path.join(wd, cid)
        if not os.path.exists(hpdb):
            run([sys.executable, args.prep_script, pdb, hpdb], os.path.join(wd, "prep.log"))
        if not os.path.exists(prefix + ".pdbqt") or not os.path.exists(prefix + ".box.txt"):
            og = next(parser.get_structure("s", pdb)[0].get_chains())[args.ser]["OG"].coord
            run([sys.executable, "-m", "meeko.cli.mk_prepare_receptor", "--read_pdb", hpdb, "-o", prefix,
                 "-p", "-v", "--box_size"] + [str(args.box_size)] * 3 +
                ["--box_center"] + [f"{x:.3f}" for x in og], os.path.join(wd, "meeko.log"))
        per_seed = {}
        for s in args.seeds:
            outp = os.path.join(wd, f"{cid}_MHET_seed{s}.pdbqt")
            if not os.path.exists(outp):
                run([args.vina, "--receptor", prefix + ".pdbqt", "--ligand", args.ligand,
                     "--config", prefix + ".box.txt", "--exhaustiveness", str(args.exhaustiveness),
                     "--num_modes", "9", "--seed", str(s), "--out", outp], os.path.join(wd, f"vina_seed{s}.log"))
            per_seed[s] = analyze(hpdb, outp)
        row = summarise(cid, per_seed)
        rows.append(row)
        consecutive_fail = 0
        print(f"[{n}/{len(pdbs)}] {cid}: best {row['best_affinity']} kcal/mol, plausible poses {row['plausible_poses']} "
              f"({time.time() - t0:.0f}s)")
    except Exception as e:
        errors.append(f"{cid}: {e}")
        consecutive_fail += 1
        print(f"[{n}/{len(pdbs)}] {cid}: ERROR - {e}")
        traceback.print_exc()
    if consecutive_fail >= 3:
        print("\nSTOPPING: 3 failures in a row. Check the .log files in the failed candidates' folders.")
        break
    if rows:
        with open(os.path.join(args.out_dir, "dock_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if errors:
    with open(os.path.join(args.out_dir, "dock_errors.txt"), "w") as f:
        f.write("\n".join(errors))
print(f"\nDone. {len(rows)} succeeded, {len(errors)} failed. Summary: {os.path.join(args.out_dir, 'dock_summary.csv')}")