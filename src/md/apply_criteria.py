import argparse, csv, glob, json, os, re
from statistics import median

ap = argparse.ArgumentParser()
ap.add_argument("--results-root", default=os.path.join("results", "md"))
ap.add_argument("--shortlist", default=os.path.join("results", "md_shortlist_final.csv"))
ap.add_argument("--out", default=os.path.join("results", "md", "criteria_table.csv"))
ap.add_argument("--merged", default=os.path.join("results", "merged_table.csv"), help="used for mutation lists if the shortlist lacks them")
args = ap.parse_args()

CLEAR, SUGGEST = 0.36, 0.18
DRIFT, RMSD, RG = 0.05, 2.7, 17.4

meta = {}
if os.path.exists(args.shortlist):
    for r in csv.DictReader(open(args.shortlist, newline="", encoding="utf-8-sig")):
        meta[r["candidate_id"].strip()] = r
    print(f"Read {len(meta)} rows from {args.shortlist}")
else:
    print(f"WARNING: shortlist not found at {os.path.abspath(args.shortlist)} - using the built-in list of arms")
merged = {}
if os.path.exists(args.merged):
    for r in csv.DictReader(open(args.merged, newline="", encoding="utf-8-sig")):
        merged[r["candidate_id"].strip()] = r.get("mutations", "")
ARM = {"rule_pick": "FoldX arm", "manual_pick": "engineering design arm", "engineering_design": "engineering design arm",
       "rational_arm": "engineering design arm", "negative_control": "negative control"}
BUILTIN = {**{c: "FoldX arm" for c in "057 070 001 002 003 005 008 009 011 019".split()},
           **{c: "negative control" for c in "047 025".split()},
           **{c: "engineering design arm" for c in "058 059 060 061 062 064 065 066".split()}}

def arm_of(cid):
    role = meta.get(cid, {}).get("role", "").strip().lower()
    if role in ARM: return ARM[role]
    return BUILTIN.get(cid.replace("candidate_", ""), "?")

def muts_of(cid):
    return meta.get(cid, {}).get("mutations", "") or merged.get(cid, "")

rows = []
for path in sorted(glob.glob(os.path.join(args.results_root, "*", "*_summary.json"))):
    s = json.load(open(path))
    name = s["name"]
    m = re.match(r"(candidate_\d+)_s(\d+)$", name)
    cid, seed = (m.group(1), int(m.group(2))) if m else (name, None)
    arm = arm_of(cid) if m else "control"
    t = s["triad_intact_fraction"]
    rating = "clearly improved" if t > CLEAR else "suggestive" if t > SUGGEST else "no evidence"
    flags = []
    if s["rmsd_drift_A_per_ns"] > DRIFT: flags.append("drift")
    if s["rmsd_mean_A"] > RMSD: flags.append("RMSD")
    if s["rg_mean_A"] > RG: flags.append("Rg")
    if arm == "control" or (seed or 1) > 1:
        advance = "-"
    else:
        advance = "second seed" if rating == "clearly improved" and not flags else ("blocked by flag" if flags and rating != "no evidence" else "-")
    rows.append({"run": name, "candidate": cid, "seed": seed, "arm": arm, "triad_intact": round(t, 3), "rating": rating,
                 "rmsd_mean_A": s["rmsd_mean_A"], "drift_A_per_ns": s["rmsd_drift_A_per_ns"], "rg_mean_A": s["rg_mean_A"],
                 "rmsf_mean_A": s["rmsf_mean_A"], "total_ns": s["total_ns"], "flags": ",".join(flags), "advance": advance,
                 "mutations": muts_of(cid)})

if not rows:
    raise SystemExit(f"No *_summary.json files found under {args.results_root}")
rows.sort(key=lambda r: -r["triad_intact"])
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

wt = [r for r in rows if r["run"].upper().startswith("WT")]
if wt:
    print(f"WT replicates (n={len(wt)}): triad-intact " + ", ".join(f"{r['triad_intact']:.0%}" for r in wt))
print(f"\n{'run':<20}{'arm':<24}{'triad':>7}  {'rating':<17}{'RMSD':>6}{'drift':>8}{'Rg':>7}  {'flags':<12}advance")
for r in rows:
    short = "incomplete" if r["total_ns"] < 29.9 else ""
    print(f"{r['run']:<20}{r['arm']:<24}{r['triad_intact']:>7.0%}  {r['rating']:<17}{r['rmsd_mean_A']:>6.2f}{r['drift_A_per_ns']:>+8.3f}{r['rg_mean_A']:>7.2f}  {r['flags'] or '-':<12}{r['advance']} {short}")

def summarize(label, group):
    if not group: return
    t = [r["triad_intact"] for r in group]
    counts = {k: sum(1 for r in group if r["rating"] == k) for k in ("clearly improved", "suggestive", "no evidence")}
    print(f"  {label:<34} n={len(group):<3} median triad {median(t):.0%}  range {min(t):.0%}-{max(t):.0%}   "
          f"clear {counts['clearly improved']}, suggestive {counts['suggestive']}, none {counts['no evidence']}")

cands = [r for r in rows if r["seed"] is not None]
by = {}
for r in cands: by.setdefault(r["candidate"], []).append(r)
cand_level = []
for c, v in by.items():
    t = sum(x["triad_intact"] for x in v) / len(v)
    cand_level.append({"candidate": c, "arm": v[0]["arm"], "mutations": v[0]["mutations"], "triad_intact": t,
                       "rating": "clearly improved" if t > CLEAR else "suggestive" if t > SUGGEST else "no evidence"})
nseeds = max((len(v) for v in by.values()), default=1)
print(f"\nGroup summaries (descriptive; one value per candidate = its mean over the seeds run so far, up to {nseeds}):")
for arm in ("FoldX arm", "engineering design arm", "negative control"):
    summarize(arm, [r for r in cand_level if r["arm"] == arm])
rat = [r for r in cand_level if r["arm"] == "engineering design arm"]
has = lambda r, m: m in r["mutations"].split(";")
print("Engineering design arm splits:")
summarize("D186V", [r for r in rat if has(r, "D186V")]); summarize("D186H", [r for r in rat if has(r, "D186H")])
a = lambda r: has(r, "A171C") and has(r, "S193C"); b = lambda r: has(r, "N233C") and has(r, "S282C")
summarize("A171C/S193C only", [r for r in rat if a(r) and not b(r)]); summarize("N233C/S282C only", [r for r in rat if b(r) and not a(r)])
summarize("both disulfide pairs", [r for r in rat if a(r) and b(r)])
summarize("I168R present", [r for r in rat if has(r, "I168R")]); summarize("I168R absent", [r for r in rat if not has(r, "I168R")])
multi = {c: sorted(v, key=lambda x: x["seed"]) for c, v in by.items() if len(v) >= 2}
tier12 = {r["candidate"] for r in cands if r["seed"] == 1 and r["advance"] in ("second seed", "blocked by flag")}
if multi:
    print("\nTwo-seed evaluation (finalist-eligible = BOTH seeds above 18% triad, neither seed flagged, AND the candidate was sent to a")
    print("second seed under the criteria or the documented deviation; second seeds of other candidates are descriptive only):")
    print(f"{'candidate':<16}{'arm':<24}{'triad by seed':<22}{'RMSD by seed':<18}{'flags':<10}eligible")
    elig = []
    for c, v in multi.items():
        ok = all(x["triad_intact"] > SUGGEST for x in v) and not any(x["flags"] for x in v)
        tri = " / ".join(f"{x['triad_intact']:.0%}" for x in v); rm = " / ".join(f"{x['rmsd_mean_A']:.2f}" for x in v)
        fl = ",".join(sorted({f for x in v for f in x["flags"].split(",") if f})) or "-"
        verdict = ("YES" if ok else "no") if c in tier12 else "descriptive only"
        print(f"{c:<16}{v[0]['arm']:<24}{tri:<22}{rm:<18}{fl:<10}{verdict}")
        if ok and c in tier12: elig.append((sum(x["triad_intact"] for x in v) / len(v), -sum(x["rmsd_mean_A"] for x in v) / len(v), c))
    elig.sort(reverse=True)
    print("Ranked finalists (mean triad, then lower mean RMSD; at most 3 go to 200 ns): " +
          (", ".join(f"{c} ({t:.0%})" for t, _, c in elig[:3]) if elig else "none yet") +
          (f"   [further eligible: {', '.join(c for _, _, c in elig[3:])}]" if len(elig) > 3 else ""))
nxt = [r["run"] for r in cands if r["advance"] == "second seed"]
print(f"\nSecond-seed candidates under the criteria ({len(nxt)}): " + (", ".join(nxt) if nxt else "none"))
print(f"Wrote {args.out}")
