import argparse, csv, glob, json, os, re
from statistics import median

ap = argparse.ArgumentParser()
ap.add_argument("--results-root", default=os.path.join("results", "md"))
ap.add_argument("--shortlist", default=os.path.join("results", "md_shortlist_final.csv"))
ap.add_argument("--out", default=os.path.join("results", "md", "criteria_table.csv"))
args = ap.parse_args()

CLEAR, SUGGEST = 0.36, 0.18
DRIFT, RMSD, RG = 0.05, 2.7, 17.4

meta = {}
if os.path.exists(args.shortlist):
    for r in csv.DictReader(open(args.shortlist, newline="")):
        meta[r["candidate_id"]] = r
ARM = {"rule_pick": "FoldX arm", "manual_pick": "rational arm", "negative_control": "negative control"}

rows = []
for path in sorted(glob.glob(os.path.join(args.results_root, "*", "*_summary.json"))):
    s = json.load(open(path))
    name = s["name"]
    m = re.match(r"(candidate_\d+)_s(\d+)$", name)
    cid, seed = (m.group(1), int(m.group(2))) if m else (name, None)
    arm = ARM.get(meta.get(cid, {}).get("role", ""), "control" if not m else "?")
    t = s["triad_intact_fraction"]
    rating = "clearly improved" if t > CLEAR else "suggestive" if t > SUGGEST else "no evidence"
    flags = []
    if s["rmsd_drift_A_per_ns"] > DRIFT: flags.append("drift")
    if s["rmsd_mean_A"] > RMSD: flags.append("RMSD")
    if s["rg_mean_A"] > RG: flags.append("Rg")
    if arm == "control":
        advance = "-"
    else:
        advance = "second seed" if rating == "clearly improved" and not flags else ("blocked by flag" if flags and rating != "no evidence" else "-")
    rows.append({"run": name, "candidate": cid, "seed": seed, "arm": arm, "triad_intact": round(t, 3), "rating": rating,
                 "rmsd_mean_A": s["rmsd_mean_A"], "drift_A_per_ns": s["rmsd_drift_A_per_ns"], "rg_mean_A": s["rg_mean_A"],
                 "rmsf_mean_A": s["rmsf_mean_A"], "total_ns": s["total_ns"], "flags": ",".join(flags), "advance": advance,
                 "mutations": meta.get(cid, {}).get("mutations", "")})

if not rows:
    raise SystemExit(f"No *_summary.json files found under {args.results_root}")
rows.sort(key=lambda r: -r["triad_intact"])
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

wt = [r for r in rows if r["run"].upper().startswith("WT")]
if wt:
    print(f"WT replicates (n={len(wt)}): triad-intact " + ", ".join(f"{r['triad_intact']:.0%}" for r in wt))
print(f"\n{'run':<20}{'arm':<17}{'triad':>7}  {'rating':<17}{'RMSD':>6}{'drift':>8}{'Rg':>7}  {'flags':<12}advance")
for r in rows:
    short = "incomplete" if r["total_ns"] < 29.9 else ""
    print(f"{r['run']:<20}{r['arm']:<17}{r['triad_intact']:>7.0%}  {r['rating']:<17}{r['rmsd_mean_A']:>6.2f}{r['drift_A_per_ns']:>+8.3f}{r['rg_mean_A']:>7.2f}  {r['flags'] or '-':<12}{r['advance']} {short}")

def summarize(label, group):
    if not group: return
    t = [r["triad_intact"] for r in group]
    counts = {k: sum(1 for r in group if r["rating"] == k) for k in ("clearly improved", "suggestive", "no evidence")}
    print(f"  {label:<34} n={len(group):<3} median triad {median(t):.0%}  range {min(t):.0%}-{max(t):.0%}   "
          f"clear {counts['clearly improved']}, suggestive {counts['suggestive']}, none {counts['no evidence']}")

cands = [r for r in rows if r["seed"] is not None]
print("\nGroup summaries (descriptive; one seed per candidate):")
for arm in ("FoldX arm", "rational arm", "negative control"):
    summarize(arm, [r for r in cands if r["arm"] == arm])
rat = [r for r in cands if r["arm"] == "rational arm"]
has = lambda r, m: m in r["mutations"].split(";")
print("Rational arm splits:")
summarize("D186V", [r for r in rat if has(r, "D186V")]); summarize("D186H", [r for r in rat if has(r, "D186H")])
a = lambda r: has(r, "A171C") and has(r, "S193C"); b = lambda r: has(r, "N233C") and has(r, "S282C")
summarize("A171C/S193C only", [r for r in rat if a(r) and not b(r)]); summarize("N233C/S282C only", [r for r in rat if b(r) and not a(r)])
summarize("both disulfide pairs", [r for r in rat if a(r) and b(r)])
summarize("I168R present", [r for r in rat if has(r, "I168R")]); summarize("I168R absent", [r for r in rat if not has(r, "I168R")])
nxt = [r["run"] for r in cands if r["advance"] == "second seed"]
print(f"\nSecond-seed candidates under the criteria ({len(nxt)}): " + (", ".join(nxt) if nxt else "none"))
print(f"Wrote {args.out}")