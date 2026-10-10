r"""
mutation_proximity.py - are each candidate's mutations near the active site, and does
that explain how far its docking score is from WT?

Usage (inside the activated 'dock' environment):
  python mutation_proximity.py --manifest C:\path\to\candidate_manifest.csv --wt-pdb C:\docking\colabfold_full\control_WT_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb --dock-summary C:\docking\dock_summary_77.csv --out C:\docking\mutation_proximity.csv
"""
import argparse, csv, re
import numpy as np
from Bio.PDB import PDBParser

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--wt-pdb", required=True)
ap.add_argument("--dock-summary", required=True)
ap.add_argument("--out", default="mutation_proximity.csv")
ap.add_argument("--offset", type=int, default=28, help="file residue = real position - offset")
ap.add_argument("--ser", type=int, default=132, help="file residue number of Ser160")
ap.add_argument("--cutoff", type=float, default=10.0, help="'near the site' distance in Angstrom")
ap.add_argument("--wt-mean", type=float, default=-4.9045, help="WT mean best affinity over 10 seeds")
ap.add_argument("--noise", type=float, default=0.15, help="WT seed-to-seed range (kcal/mol)")
args = ap.parse_args()

THREE2ONE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
             "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
split = lambda s: [x for x in re.split(r"[;,\s]+", (s or "").strip()) if x]

chain = next(PDBParser(QUIET=True).get_structure("wt", args.wt_pdb)[0].get_chains())
og = chain[args.ser]["OG"].coord
def res_dist(pos):
    res = chain[pos - args.offset]
    return THREE2ONE.get(res.get_resname(), "?"), min(float(np.linalg.norm(a.coord - og)) for a in res)

dock = {r["candidate_id"]: float(r["mean_best_affinity"]) for r in csv.DictReader(open(args.dock_summary))}

rows = []
for r in csv.DictReader(open(args.manifest)):
    cid = r["candidate_id"].strip()
    wts, muts, poss = split(r["wild_type"]), split(r["mutation_type"]), [int(p) for p in split(r["position_idx"])]
    items = []
    for w, m, p in sorted(zip(wts, muts, poss), key=lambda t: t[2]):
        aa, d = res_dist(p)
        if aa != w:
            print(f"WARNING {cid}: position {p} is {aa} in the WT model but manifest says {w}")
        items.append((f"{w}{p}{m}", d))
    near = [(lab, d) for lab, d in items if d <= args.cutoff]
    nearest = min(items, key=lambda t: t[1])
    delta = dock[cid] - args.wt_mean if cid in dock else None
    rows.append({
        "candidate_id": cid, "n_mut": len(items), "n_near": len(near),
        "nearest_mutation": f"{nearest[0]} ({nearest[1]:.1f} A)",
        "near_mutations": ";".join(f"{lab}({d:.1f})" for lab, d in near),
        "mean_best_affinity": dock.get(cid, ""), "delta_vs_WT": None if delta is None else round(delta, 2),
        "touches_90_or_168": any(p in (90, 168) for p in poss),
    })

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"\nCandidates with their docking shift (WT mean {args.wt_mean}; noise range +/-{args.noise / 2:.2f}); biggest shifts first")
print(f"{'id':<15}{'delta':>7} {'near':>5}  mutations within {args.cutoff:.0f} A of Ser160 OG  |  nearest")
for r in sorted([x for x in rows if x["delta_vs_WT"] is not None], key=lambda x: -abs(x["delta_vs_WT"]))[:12]:
    print(f"{r['candidate_id']:<15}{r['delta_vs_WT']:>7} {r['n_near']:>5}  {r['near_mutations'] or '-'}  |  {r['nearest_mutation']}")

have = [x for x in rows if x["delta_vs_WT"] is not None]
for label, grp in (("no mutation near the site", [x for x in have if x["n_near"] == 0]),
                   ("1 or more mutations near the site", [x for x in have if x["n_near"] > 0])):
    if grp:
        d = np.array([abs(x["delta_vs_WT"]) for x in grp])
        print(f"\n{label}: {len(grp)} candidates, mean |shift| {d.mean():.3f}, largest {d.max():.2f}, "
              f"{int((d > args.noise).sum())} beyond the noise range")
print(f"\nWrote {args.out}")
