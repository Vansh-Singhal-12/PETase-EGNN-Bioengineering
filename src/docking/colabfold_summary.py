r"""
colabfold_summary.py - summarise the ColabFold rank_001 models for every candidate.

Usage (inside the activated 'dock' environment):
  python colabfold_summary.py --pdb-dir C:\docking\structures --json-dir C:\docking\scores --out colabfold_summary.csv

For each candidate it reports: mean pLDDT, pTM, pLDDT around the active site, C-alpha RMSD
to the WT ColabFold model (whole protein and active-site region), and two triad distances.
"""
import argparse, csv, glob, json, os, re
import numpy as np
from Bio.PDB import PDBParser, Superimposer

ap = argparse.ArgumentParser()
ap.add_argument("--pdb-dir", required=True)
ap.add_argument("--json-dir", required=True)
ap.add_argument("--out", default="colabfold_summary.csv")
ap.add_argument("--wt-id", default="control_WT")
ap.add_argument("--ser", type=int, default=132, help="file residue number of Ser160")
ap.add_argument("--asp", type=int, default=178, help="file residue number of Asp206")
ap.add_argument("--his", type=int, default=209, help="file residue number of His237")
ap.add_argument("--radius", type=float, default=10.0, help="active-site region radius (Angstrom)")
args = ap.parse_args()

parser = PDBParser(QUIET=True)

def load(pdb_path):
    chain = next(parser.get_structure("s", pdb_path)[0].get_chains())
    return chain

def region_residues(chain, centre, radius):
    out = []
    for res in chain:
        if any(np.linalg.norm(a.coord - centre) <= radius for a in res):
            out.append(res.id[1])
    return out

models = {}
for p in sorted(glob.glob(os.path.join(args.pdb_dir, "*_unrelaxed_rank_001_*.pdb"))):
    cid = os.path.basename(p).split("_unrelaxed_rank_001")[0]
    models[cid] = p
if args.wt_id not in models:
    raise SystemExit(f"WT model '{args.wt_id}' not found in {args.pdb_dir}")

wt_chain = load(models[args.wt_id])
wt_ca = {r.id[1]: r["CA"] for r in wt_chain}
wt_og = wt_chain[args.ser]["OG"].coord
region = region_residues(wt_chain, wt_og, args.radius)
print(f"Active-site region: {len(region)} residues within {args.radius} A of Ser OG in the WT model")

def triad(chain):
    og = chain[args.ser]["OG"].coord
    ne2 = chain[args.his]["NE2"].coord
    nd1 = chain[args.his]["ND1"].coord
    od = min(np.linalg.norm(nd1 - chain[args.asp][n].coord) for n in ("OD1", "OD2"))
    return float(np.linalg.norm(og - ne2)), float(od)

rows = []
for cid, p in models.items():
    chain = load(p)
    jpath = glob.glob(os.path.join(args.json_dir, f"{cid}_scores_rank_001_*.json"))
    plddt_list, ptm = None, None
    if jpath:
        j = json.load(open(jpath[0]))
        plddt_list = j.get("plddt"); ptm = j.get("ptm")
    resnums = [r.id[1] for r in chain]
    if plddt_list is None:                                  # fall back to B-factors
        plddt_list = [r["CA"].get_bfactor() for r in chain]
    plddt = dict(zip(resnums, plddt_list))

    ca = {r.id[1]: r["CA"] for r in chain}
    common = sorted(set(ca) & set(wt_ca))
    sup = Superimposer()
    sup.set_atoms([wt_ca[i] for i in common], [ca[i] for i in common])
    rot, tran = sup.rotran
    glob_rmsd = float(sup.rms)
    # active-site region RMSD using the global alignment
    dev = [np.linalg.norm(wt_ca[i].coord - (np.dot(ca[i].coord, rot) + tran)) for i in region if i in ca]
    site_rmsd = float(np.sqrt(np.mean(np.square(dev))))

    d_ser_his, d_his_asp = triad(chain)
    rows.append({
        "candidate_id": cid,
        "mean_plddt": round(float(np.mean(list(plddt.values()))), 1),
        "ptm": None if ptm is None else round(float(ptm), 3),
        "site_plddt": round(float(np.mean([plddt[i] for i in region if i in plddt])), 1),
        "rmsd_to_WT_all": round(glob_rmsd, 2),
        "rmsd_to_WT_site": round(site_rmsd, 2),
        "SerOG_HisNE2": round(d_ser_his, 2),
        "HisND1_AspOD": round(d_his_asp, 2),
    })

def key(r):
    m = re.search(r"(\d+)$", r["candidate_id"])
    return (0, int(m.group(1))) if m and r["candidate_id"].startswith("candidate") else (1, r["candidate_id"])
rows.sort(key=key)

for r in rows:
    flags = []
    if r["mean_plddt"] < 70: flags.append("LOW_PLDDT")
    if r["site_plddt"] < 80: flags.append("LOW_SITE_PLDDT")
    if r["rmsd_to_WT_all"] > 1.0: flags.append("RMSD>1A")
    r["flags"] = ";".join(flags)

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"{'id':<22}{'pLDDT':>7}{'site':>7}{'RMSD':>7}{'siteRMSD':>9}{'S-H':>6}{'H-D':>6}  flags")
for r in rows:
    print(f"{r['candidate_id']:<22}{r['mean_plddt']:>7}{r['site_plddt']:>7}{r['rmsd_to_WT_all']:>7}"
          f"{r['rmsd_to_WT_site']:>9}{r['SerOG_HisNE2']:>6}{r['HisND1_AspOD']:>6}  {r['flags']}")
print(f"\nWrote {args.out} ({len(rows)} models)")
