import argparse, csv, json

ap = argparse.ArgumentParser()
ap.add_argument("--criteria", required=True)
ap.add_argument("--inputs-queue", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tier", type=int, default=3)
ap.add_argument("--seed", type=int, default=2)
args = ap.parse_args()

pdb_of = {j["name"][:-3]: j["pdb"] for j in json.load(open(args.inputs_queue)) if j["name"].endswith("_s1")}
rows = [r for r in csv.DictReader(open(args.criteria, newline="")) if r["seed"] == "1"]
tiers = []
for r in rows:
    if r["advance"] == "second seed": t = 1
    elif r["advance"] == "blocked by flag": t = 2
    else: t = 3
    tiers.append((t, -float(r["triad_intact"]), r["candidate"], r["arm"]))
tiers.sort()
jobs, shown = [], {1: [], 2: [], 3: []}
for t, neg, cid, arm in tiers:
    if t > args.max_tier: continue
    if cid not in pdb_of:
        print(f"WARNING: {cid} not in the inputs queue - skipped"); continue
    jobs.append({"name": f"{cid}_s{args.seed}", "pdb": pdb_of[cid], "seed": args.seed, "ns": 30, "role": f"{arm} (tier {t})", "tier": t})
    shown[t].append(f"{cid} ({-neg:.0%})")
json.dump(jobs, open(args.out, "w"), indent=1)
labels = {1: "tier 1 (criteria)", 2: "tier 2 (documented deviation)", 3: "tier 3 (replication of the rest)"}
for t in (1, 2, 3):
    if shown[t]: print(f"{labels[t]}: " + ", ".join(shown[t]))
print(f"\n{len(jobs)} second-seed jobs written to {args.out}")
