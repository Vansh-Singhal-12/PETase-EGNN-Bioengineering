import argparse, csv, random
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--merged", required=True)
ap.add_argument("--out", default="md_shortlist.csv")
ap.add_argument("--n-picks", type=int, default=10, help="rule-based picks (manual ones are extra)")
ap.add_argument("--bg-frac", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--manual", action="append", default=[], help='ID:reason, may be repeated')
args = ap.parse_args()

def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

rows = list(csv.DictReader(open(args.merged, newline="")))
for r in rows:
    r["muts"] = set(r["mutations"].split(";"))
    r["foldx"], r["esm"], r["drift"] = f(r["foldx_score"]), f(r["esm_additive_score"]), f(r["max_triad_embedding_drift"])
    r["n"] = len(r["muts"])
    r["cys"] = sum(m.endswith("C") for m in r["muts"])
n = len(rows)
freq = Counter(m for r in rows for m in r["muts"])
background = {m for m, c in freq.items() if c / n >= args.bg_frac}
for r in rows:
    r["dist"] = r["muts"] - background
print(f"{n} candidates. Background mutations (in >= {args.bg_frac:.0%} of the pool): "
      + ", ".join(f"{m} ({freq[m]})" for m in sorted(background, key=lambda m: -freq[m])))

ranked = sorted([r for r in rows if r["foldx"] is not None], key=lambda r: -r["foldx"])
picked, reasons, relaxed_notes = [], {}, []

def min_diff(r):
    return min((len(r["dist"] ^ p["dist"]) for p in picked), default=99)

def try_add(r, why):
    if r["candidate_id"] in reasons:
        reasons[r["candidate_id"]].append(why); return True
    for need in (3, 2, 1):
        if min_diff(r) >= need:
            picked.append(r); reasons[r["candidate_id"]] = [why]
            if need < 3: relaxed_notes.append(f"{r['candidate_id']} ({why}) added with diversity threshold {need}")
            return True
    return False

def best(pred):
    return next((r for r in ranked if pred(r)), None)

slots = [("disulfide design", lambda r: r["cys"] >= 2),
         ("5-point combination", lambda r: r["n"] == 5),
         ("lacks E231K", lambda r: "E231K" not in r["muts"]),
         ("ESM consensus (esm > 0)", lambda r: r["esm"] is not None and r["esm"] > 0)]
for why, pred in slots:
    cand = [r for r in ranked if pred(r)]
    if not cand:
        print(f"NOTE: no candidate for slot '{why}'"); continue
    for r in cand:                              # best FoldX one that satisfies diversity (or is already picked)
        if r["candidate_id"] in reasons or min_diff(r) >= 1:
            try_add(r, why); break

for r in ranked:                                # fill the rest in FoldX order
    if len(picked) >= args.n_picks: break
    if r["candidate_id"] not in reasons:
        try_add(r, "FoldX rank")

manual = []
for item in args.manual:
    cid, _, why = item.partition(":")
    r = next((x for x in rows if x["candidate_id"] == cid), None)
    if r is None: print(f"WARNING: manual id {cid} not found"); continue
    if cid in reasons: print(f"NOTE: {cid} was already picked by the rules"); continue
    manual.append(r); reasons[cid] = [f"MANUAL: {why}"]

pool_med = sorted(r["foldx"] for r in ranked)[len(ranked) // 2]
chosen_ids = set(reasons)
negs = [r for r in rows if r["candidate_id"] not in chosen_ids and r["foldx"] is not None
        and abs(r["foldx"] - pool_med) <= 3 and 7 <= r["n"] <= 9]
random.Random(args.seed).shuffle(negs)
negs = negs[:2]

print(f"\nPool median FoldX score: {pool_med:.1f}")
print(f"\nPROPOSED MD SET ({len(picked)} rule-based + {len(manual)} manual + {len(negs)} negative controls + controls)\n")
print(f"{'id':<15}{'n':>3}{'foldx':>7}{'esm':>7}{'drift':>7} {'flags':<14}distinguishing mutations   <- why")
def show(r, why):
    fl = []
    if r["drift_flag"] == "True": fl.append("drift")
    if r["touches_90_or_168"] == "True": fl.append("90/168")
    esm = "" if r["esm"] is None else f"{r['esm']:.1f}"
    dr = "" if r["drift"] is None else f"{r['drift']:.2f}"
    print(f"{r['candidate_id']:<15}{r['n']:>3}{r['foldx']:>7.1f}{esm:>7}{dr:>7} {','.join(fl) or '-':<14}"
          f"{','.join(sorted(r['dist'])) or '(none)'}   <- {why}")
for r in picked + manual:
    show(r, "; ".join(reasons[r["candidate_id"]]))
for r in negs:
    show(r, "NEGATIVE CONTROL (median FoldX)")
print("Fixed controls: control_WT (3 replicates), control_FASTPETase")
for note in relaxed_notes:
    print("NOTE:", note)

with open(args.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["candidate_id", "role", "reason", "n_mut", "foldx_score", "esm_additive_score", "drift_flag", "touches_90_or_168", "mutations"])
    for r in picked: w.writerow([r["candidate_id"], "rule_pick", "; ".join(reasons[r["candidate_id"]]), r["n"], r["foldx"], r["esm"], r["drift_flag"], r["touches_90_or_168"], r["mutations"]])
    for r in manual: w.writerow([r["candidate_id"], "manual_pick", "; ".join(reasons[r["candidate_id"]]), r["n"], r["foldx"], r["esm"], r["drift_flag"], r["touches_90_or_168"], r["mutations"]])
    for r in negs: w.writerow([r["candidate_id"], "negative_control", "median FoldX", r["n"], r["foldx"], r["esm"], r["drift_flag"], r["touches_90_or_168"], r["mutations"]])
    w.writerow(["control_WT", "control", "3 replicates, seeds 1-3", "", "", "", "", "", ""])
    w.writerow(["control_FASTPETase", "control", "known thermostable variant", 5, "", "", "", "", "S121E;D186H;R224Q;N233K;R280A"])
print(f"\nWrote {args.out}")