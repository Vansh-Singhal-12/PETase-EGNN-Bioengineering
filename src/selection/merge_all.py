import argparse, csv, re
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--shortlist", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--colabfold", required=True)
ap.add_argument("--dock", required=True)
ap.add_argument("--proximity", default="")
ap.add_argument("--out", default="merged_table.csv")
args = ap.parse_args()

split = lambda s: [x for x in re.split(r"[;,\s]+", (s or "").strip()) if x]
def key_of(row):
    wts, muts, poss = split(row["wild_type"]), split(row["mutation_type"]), [int(p) for p in split(row["position_idx"])]
    return tuple(sorted(zip(poss, wts, muts)))
def label(key):
    return ";".join(f"{w}{p}{m}" for p, w, m in key)
num = lambda s: float(s) if str(s).strip() not in ("", "None") else None
rd = lambda path: list(csv.DictReader(open(path, newline="")))

manifest = {key_of(r): r["candidate_id"].strip() for r in rd(args.manifest)}
cf = {r["candidate_id"]: r for r in rd(args.colabfold)}
dk = {r["candidate_id"]: r for r in rd(args.dock)}
px = {r["candidate_id"]: r for r in rd(args.proximity)} if args.proximity else {}

rows, unmatched = [], []
for r in rd(args.shortlist):
    k = key_of(r)
    cid = manifest.get(k)
    if cid is None:
        unmatched.append(label(k)); continue
    fx = num(r.get("foldx_large_score", ""))
    if fx is None:
        fx = num(r.get("foldx_quint_score", ""))
    positions = [p for p, _, _ in k]
    rows.append({
        "candidate_id": cid, "n_mut": len(k), "mutations": label(k),
        "foldx_score": fx,
        "esm_additive_score": num(r.get("esm_additive_score", "")),
        "max_triad_embedding_drift": num(r.get("max_triad_embedding_drift", "")),
        "drift_flag": r.get("drift_flag", ""),
        "is_flexible_region": r.get("is_flexible_region", ""),
        "touches_90_or_168": any(p in (90, 168) for p in positions),
        "category": r.get("category", ""),
        "epistasis_info_only": num(r.get("epistasis_3singles", "")) if r.get("epistasis_3singles", "") != "" else num(r.get("foldx_quint_epistasis", "")),
        "mean_plddt": cf.get(cid, {}).get("mean_plddt", ""),
        "rmsd_to_WT_all": cf.get(cid, {}).get("rmsd_to_WT_all", ""),
        "dock_best_affinity": dk.get(cid, {}).get("best_affinity", ""),
        "dock_plausible_poses": dk.get(cid, {}).get("plausible_poses", ""),
        "n_mut_near_site": px.get(cid, {}).get("n_near", ""),
    })

if unmatched:
    print(f"WARNING: {len(unmatched)} shortlist rows matched no manifest entry (first 3):")
    for u in unmatched[:3]: print("   ", u)
missing = set(manifest.values()) - {r["candidate_id"] for r in rows}
if missing:
    print(f"WARNING: {len(missing)} manifest candidates not found in the shortlist:", sorted(missing)[:5])
dupes = [c for c, n in Counter(r["candidate_id"] for r in rows).items() if n > 1]
if dupes:
    print("WARNING: shortlist rows matching the same candidate:", dupes)

rows.sort(key=lambda r: r["candidate_id"])
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"\nMerged {len(rows)} candidates -> {args.out}")
fx = sorted(r["foldx_score"] for r in rows if r["foldx_score"] is not None)
if fx:
    print(f"FoldX score (positive = stabilizing): min {fx[0]:.1f}, median {fx[len(fx)//2]:.1f}, max {fx[-1]:.1f}; "
          f"{sum(r['foldx_score'] is None for r in rows)} missing")
print("Mutation counts:", dict(sorted(Counter(r["n_mut"] for r in rows).items())))
print("Categories:", dict(Counter(r["category"] or "(blank)" for r in rows)))
print("drift_flag:", dict(Counter(r["drift_flag"] or "(blank)" for r in rows)),
      "| touching position 90 or 168:", sum(r["touches_90_or_168"] for r in rows))

print("\nTop 15 by FoldX score:")
print(f"{'id':<15}{'n':>3}{'foldx':>8}{'esm':>8}{'drift':>7} {'drift_flag':<11}{'90/168':<7}{'dock':>7}  category")
for r in sorted([x for x in rows if x["foldx_score"] is not None], key=lambda x: -x["foldx_score"])[:15]:
    esm = "" if r["esm_additive_score"] is None else f"{r['esm_additive_score']:.1f}"
    dr = "" if r["max_triad_embedding_drift"] is None else f"{r['max_triad_embedding_drift']:.2f}"
    print(f"{r['candidate_id']:<15}{r['n_mut']:>3}{r['foldx_score']:>8.1f}{esm:>8}{dr:>7} {str(r['drift_flag']):<11}"
          f"{str(r['touches_90_or_168']):<7}{r['dock_best_affinity']:>7}  {r['category']}")

common = Counter(m for r in rows for m in r["mutations"].split(";"))
print("\nMost shared mutations (how similar the candidates are to each other):")
print("  " + ", ".join(f"{m} x{n}" for m, n in common.most_common(12)))