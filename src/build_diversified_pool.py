import argparse
import csv

BULKY_HYDROPHOBIC = set("ILVFWMY")
SMALL_FLEXIBLE = set("GPAS")

# Positions with real, literature-verified experimental data in the
# training set (Stevensen et al., Rennison et al.) -- force-included
# regardless of EGNN rank, since these are the only ground-truth-anchored
# positions and Stevensen et al. specifically documented non-additive
# (epistatic) combo effects at some of these.
DEFAULT_HOTSPOTS = {117, 119, 136, 168, 188}


def is_bulky_to_small(wt, mut):
    return wt in BULKY_HYDROPHOBIC and mut in SMALL_FLEXIBLE


def build_pool(ranked_csv_path, output_path, pool_size=400, bulky_small_cap_frac=0.40,
               bias_check_n=8, hotspots=None):
    if hotspots is None:
        hotspots = DEFAULT_HOTSPOTS

    with open(ranked_csv_path) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["_pos"] = int(r["position_idx"].split(",")[0])
        r["_is_bulky_small"] = is_bulky_to_small(r["wild_type"], r["mutation_type"])

    # 1. Hotspots -- force include ALL rows at these positions, regardless of rank
    hotspot_rows = [r for r in rows if r["_pos"] in hotspots]

    # 2. Bias-check subset -- top N bulky->small candidates, kept separate and
    # explicitly labeled
    bulky_small_sorted = [r for r in rows if r["_is_bulky_small"]]
    bias_check_rows = bulky_small_sorted[:bias_check_n]
    bias_check_keys = {(r["_pos"], r["mutation_type"]) for r in bias_check_rows}

    # 3. Diversified pool -- walk the full ranked list top-down, capping how much
    # of the pool can be bulky->small so the pattern can't dominate.
    max_bulky_small = int(pool_size * bulky_small_cap_frac)
    pool_rows, n_bulky_small_added = [], 0
    already_included_keys = {(r["_pos"], r["mutation_type"]) for r in hotspot_rows} | bias_check_keys

    for r in rows:
        if len(pool_rows) >= pool_size:
            break
        key = (r["_pos"], r["mutation_type"])
        if key in already_included_keys:
            continue
        if r["_is_bulky_small"]:
            if n_bulky_small_added >= max_bulky_small:
                continue
            n_bulky_small_added += 1
        pool_rows.append(r)
        already_included_keys.add(key)

    def tag_and_write(rows_list, category):
        out = []
        for r in rows_list:
            out.append({
                "wild_type": r["wild_type"], "mutation_type": r["mutation_type"],
                "position_idx": r["position_idx"], "predicted_score": r["predicted_score"],
                "category": category,
            })
        return out

    all_out = (tag_and_write(hotspot_rows, "hotspot")
               + tag_and_write(bias_check_rows, "bias_check")
               + tag_and_write(pool_rows, "diversified_pool"))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "predicted_score", "category"])
        writer.writeheader()
        writer.writerows(all_out)

    print(f"[build_pool] Hotspot rows (force-included): {len(hotspot_rows)}")
    print(f"[build_pool] Bias-check rows (bulky->small, top {bias_check_n}): {len(bias_check_rows)}")
    print(f"[build_pool] Diversified pool: {len(pool_rows)} "
          f"({n_bulky_small_added} bulky->small, capped at {max_bulky_small} = {bulky_small_cap_frac*100:.0f}%)")
    print(f"[build_pool] Total unique candidates: {len(all_out)} -> {output_path}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_csv", type=str, default="results/round1_ranked.csv")
    ap.add_argument("--output", type=str, default="results/diversified_pool.csv")
    ap.add_argument("--pool_size", type=int, default=400)
    ap.add_argument("--bulky_small_cap_frac", type=float, default=0.40)
    ap.add_argument("--bias_check_n", type=int, default=8)
    args = ap.parse_args()

    build_pool(args.ranked_csv, args.output, pool_size=args.pool_size,
               bulky_small_cap_frac=args.bulky_small_cap_frac, bias_check_n=args.bias_check_n)