"""
Supplements an existing shortlist with N more single-point candidates whose
WILD-TYPE residue is outside the bulky-hydrophobic family entirely (not just
filtered by mutation-type pattern, which still let residues like F/M/Y/W/L
dominate the tail of the previous fill). This guarantees the added rows
represent genuinely different chemistry, not just a looser version of the
same bias.

Respects the existing shortlist's per-position usage -- won't push any
position over the same cap used when the shortlist was built.
"""
import argparse
import csv
from collections import Counter

# Broader exclusion than the strict pattern check: ANY wild-type in this set
# is skipped entirely, regardless of what it's mutating to, since these are
# exactly the residue types driving the observed bias.
BULKY_HYDROPHOBIC_FAMILY = set("ILVFWMYR")


def load_position_counts(shortlist_path):
    counts = Counter()
    with open(shortlist_path) as f:
        for r in csv.DictReader(f):
            for p in r["position_idx"].split(","):
                counts[int(p)] += 1
    return counts


def add_diverse_singles(shortlist_path, round1_ranked_path, output_path,
                         n_to_add=15, max_per_position=3):
    position_counts = load_position_counts(shortlist_path)

    existing_rows = list(csv.DictReader(open(shortlist_path)))
    existing_keys = {(r["wild_type"], r["mutation_type"], r["position_idx"]) for r in existing_rows}

    with open(round1_ranked_path) as f:
        candidates = list(csv.DictReader(f))
    candidates.sort(key=lambda r: float(r["predicted_score"]), reverse=True)

    added = []
    for r in candidates:
        if len(added) >= n_to_add:
            break
        wt = r["wild_type"]
        if wt in BULKY_HYDROPHOBIC_FAMILY:
            continue  # the whole point of this pass -- skip regardless of mutation type

        pos = int(r["position_idx"])
        if position_counts[pos] >= max_per_position:
            continue

        key = (r["wild_type"], r["mutation_type"], r["position_idx"])
        if key in existing_keys:
            continue

        added.append({
            "wild_type": r["wild_type"], "mutation_type": r["mutation_type"],
            "position_idx": r["position_idx"], "combo_score": r["predicted_score"],
            "epistasis_score": "", "category": "single_diverse_wt",
        })
        existing_keys.add(key)
        position_counts[pos] += 1

    all_rows = existing_rows + added

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "combo_score", "epistasis_score", "category"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[add_diverse_singles] Added {len(added)} candidates (target was {n_to_add})")
    if added:
        print(f"[add_diverse_singles] Wild-types represented in the addition: "
              f"{sorted(set(r['wild_type'] for r in added))}")
    print(f"[add_diverse_singles] TOTAL shortlist size: {len(all_rows)} -> {output_path}")

    if len(added) < n_to_add:
        print(f"\n[add_diverse_singles] WARNING: only found {len(added)}/{n_to_add} candidates matching "
              f"the diversity + cap constraints. This could mean genuinely non-bulky wild-types are rare "
              f"among high-scoring predictions -- worth checking round1_ranked.csv manually if this number "
              f"is much lower than requested.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", type=str, default="results/foldx_shortlist.csv")
    ap.add_argument("--round1_ranked", type=str, default="results/round1_ranked.csv")
    ap.add_argument("--output", type=str, default="results/foldx_shortlist.csv")
    ap.add_argument("--n_to_add", type=int, default=15)
    ap.add_argument("--max_per_position", type=int, default=3)
    args = ap.parse_args()

    add_diverse_singles(args.shortlist, args.round1_ranked, args.output,
                         n_to_add=args.n_to_add, max_per_position=args.max_per_position)