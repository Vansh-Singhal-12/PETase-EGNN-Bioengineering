"""
Builds the final FoldX/ColabFold shortlist (~100 candidates) from four
sources, with a GLOBAL per-position cap enforced across all sources
together -- prevents a position that already appears 3x via one source
(e.g. epistasis) from sneaking back in via another (e.g. raw combo score),
which would defeat the whole point of capping.

Sources, in priority order (hotspot-anchored candidates get first claim on
their position's cap slots, since they're the only ones anchored to real
experimental data):
  1. Hotspot-anchored combos  -- any combo touching a literature-verified
     position (117, 119, 136, 168, 188), ranked by combo_score
  2. Epistasis-ranked combos  -- top by epistasis_score
  3. Raw-score combos         -- top by combo_score
  4. Bias-check combos        -- deliberately chosen worst-offender
     bulky->small/buried picks, EXEMPT from the position cap (the whole
     point is to re-test the same positions FoldX already saw fail)
  5. Diversified singles      -- fills any remaining slots with top
     single-point candidates from the Round 1 diversified pool, for
     single-mutation coverage in the FoldX batch too
"""
import argparse
import csv
from collections import Counter

DEFAULT_HOTSPOTS = {117, 119, 136, 168, 188}


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get_positions(row):
    return [int(p) for p in row["position_idx"].split(",")]


def touches_hotspot(row, hotspots):
    return any(p in hotspots for p in get_positions(row))


def build_shortlist(round1_ranked_path, round2_combo_path, diversified_pool_path,
                     output_path,
                     n_hotspot=25, n_epistasis=25, n_raw_combo=20,
                     n_bias_check=10, n_singles=20,
                     max_per_position=3, hotspots=None):
    if hotspots is None:
        hotspots = DEFAULT_HOTSPOTS

    combo_rows = load_rows(round2_combo_path)
    combo_rows = [r for r in combo_rows if r.get("combo_score")]
    for r in combo_rows:
        r["_positions"] = get_positions(r)

    pool_rows = load_rows(diversified_pool_path)

    position_count = Counter()
    selected = []
    selected_keys = set()

    def row_key(row, kind):
        return (kind, row.get("wild_type"), row.get("mutation_type"), row.get("position_idx"))

    def can_add(row, respect_cap=True):
        if not respect_cap:
            return True
        for p in row["_positions"]:
            if position_count[p] >= max_per_position:
                return False
        return True

    def add_row(row, category, respect_cap=True):
        key = row_key(row, "combo")
        if key in selected_keys:
            return False
        if not can_add(row, respect_cap=respect_cap):
            return False
        selected.append({
            "wild_type": row["wild_type"], "mutation_type": row["mutation_type"],
            "position_idx": row["position_idx"], "combo_score": row.get("combo_score", ""),
            "epistasis_score": row.get("epistasis_score", ""), "category": category,
        })
        selected_keys.add(key)
        for p in row["_positions"]:
            position_count[p] += 1
        return True

    # 1. Hotspot-anchored combos (priority -- real data anchor)
    hotspot_combos = [r for r in combo_rows if touches_hotspot(r, hotspots)]
    hotspot_combos.sort(key=lambda r: float(r["combo_score"]), reverse=True)
    n_added = 0
    for r in hotspot_combos:
        if n_added >= n_hotspot:
            break
        if add_row(r, "hotspot_combo"):
            n_added += 1
    print(f"[shortlist] Hotspot-anchored combos added: {n_added}/{n_hotspot}")

    # 2. Epistasis-ranked combos
    epistasis_rows = [r for r in combo_rows if r.get("epistasis_score")]
    epistasis_rows.sort(key=lambda r: float(r["epistasis_score"]), reverse=True)
    n_added = 0
    for r in epistasis_rows:
        if n_added >= n_epistasis:
            break
        if add_row(r, "epistasis"):
            n_added += 1
    print(f"[shortlist] Epistasis-ranked combos added: {n_added}/{n_epistasis}")

    # 3. Raw combo score
    raw_sorted = sorted(combo_rows, key=lambda r: float(r["combo_score"]), reverse=True)
    n_added = 0
    for r in raw_sorted:
        if n_added >= n_raw_combo:
            break
        if add_row(r, "raw_combo_score"):
            n_added += 1
    print(f"[shortlist] Raw-score combos added: {n_added}/{n_raw_combo}")

    # 4. Bias-check -- deliberately the worst offenders, EXEMPT from cap
    # (re-test the same positions here to confirm FoldX rejects them)
    n_added = 0
    for r in raw_sorted:
        if n_added >= n_bias_check:
            break
        key = row_key(r, "combo")
        if key in selected_keys:
            continue
        if add_row(r, "bias_check", respect_cap=False):
            n_added += 1
    print(f"[shortlist] Bias-check combos added (cap-exempt): {n_added}/{n_bias_check}")

    # 5. Fill remaining slots with diversified single-point candidates
    n_added = 0
    for r in pool_rows:
        if n_added >= n_singles:
            break
        pos = int(r["position_idx"].split(",")[0])
        if position_count[pos] >= max_per_position:
            continue
        key = ("single", r["wild_type"], r["mutation_type"], r["position_idx"])
        if key in selected_keys:
            continue
        selected.append({
            "wild_type": r["wild_type"], "mutation_type": r["mutation_type"],
            "position_idx": r["position_idx"], "combo_score": r.get("predicted_score", ""),
            "epistasis_score": "", "category": f"single_{r.get('category', 'pool')}",
        })
        selected_keys.add(key)
        position_count[pos] += 1
        n_added += 1
    print(f"[shortlist] Diversified singles added: {n_added}/{n_singles}")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "combo_score", "epistasis_score", "category"])
        writer.writeheader()
        writer.writerows(selected)

    print(f"\n[shortlist] TOTAL: {len(selected)} candidates -> {output_path}")
    print(f"\n[shortlist] Position usage (top 15 most-used, cap={max_per_position} except bias_check):")
    for pos, count in position_count.most_common(15):
        print(f"    Position {pos:<5}: {count}x")
    print(f"\n[shortlist] Unique positions touched: {len(position_count)}")

    category_counts = Counter(r["category"] for r in selected)
    print(f"\n[shortlist] Breakdown by category:")
    for cat, count in category_counts.most_common():
        print(f"    {cat:<20}: {count}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1_ranked", type=str, default="results/round1_ranked.csv")
    ap.add_argument("--round2_combo", type=str, default="results/round2_combo_ranked.csv")
    ap.add_argument("--diversified_pool", type=str, default="results/diversified_pool.csv")
    ap.add_argument("--output", type=str, default="results/foldx_shortlist.csv")
    ap.add_argument("--n_hotspot", type=int, default=25)
    ap.add_argument("--n_epistasis", type=int, default=25)
    ap.add_argument("--n_raw_combo", type=int, default=20)
    ap.add_argument("--n_bias_check", type=int, default=10)
    ap.add_argument("--n_singles", type=int, default=20)
    ap.add_argument("--max_per_position", type=int, default=3)
    args = ap.parse_args()

    build_shortlist(args.round1_ranked, args.round2_combo, args.diversified_pool, args.output,
                     n_hotspot=args.n_hotspot, n_epistasis=args.n_epistasis,
                     n_raw_combo=args.n_raw_combo, n_bias_check=args.n_bias_check,
                     n_singles=args.n_singles, max_per_position=args.max_per_position)