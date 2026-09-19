"""
Selects the validated pairwise combo pool from combo_master_table.csv.

Fixes applied after review:
- touches_caution is now written to output (was computed but dropped).
- Each category (top_foldx_score, top_epistasis) gets its OWN position
  counter, not a shared one -- prevents the first pass from starving the
  second pass's access to a position via a shared cap.
- Soft deprioritization: within each category, non-caution candidates are
  filled first; caution-flagged (positions 90/168) candidates are only
  added if the quota isn't met otherwise. Still not a hard exclusion --
  FoldX being wrong about I168R's direction doesn't mean position 168
  itself is bad, just that FoldX's specific number there isn't trustworthy.
"""
import argparse
import csv
from collections import Counter

CAUTION_POSITIONS = {90, 168}

KNOWN_REAL_COMBOS = {
    frozenset({("I", "R", 168), ("S", "Q", 188)}),  # Stevensen et al., real epistasis ~ -0.04
}


def build_pool(master_table_path, output_path, n_top_score=40, n_top_epistasis=25,
               max_per_position=4, min_combo_score=1.5):
    with open(master_table_path) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["_score"] = float(r["foldx_combo_score"]) if r.get("foldx_combo_score") else None
        r["_epistasis"] = float(r["foldx_epistasis"]) if r.get("foldx_epistasis") else None
        r["_pos"] = [int(p) for p in r["position_idx"].split(",")]
        r["_touches_caution"] = any(p in CAUTION_POSITIONS for p in r["_pos"])

    selected = []
    selected_keys = set()

    def matches_known_combo(row):
        wt_parts = row["wild_type"].split(";")
        mut_parts = row["mutation_type"].split(";")
        row_muts = frozenset(zip(wt_parts, mut_parts, row["_pos"]))
        return row_muts in KNOWN_REAL_COMBOS

    def fill_category(candidates, quota, category, position_counter):
        """Two sub-passes: non-caution candidates first, then caution-flagged
        ones only if quota isn't otherwise met. Each category uses its OWN
        position_counter, passed in fresh -- no cross-category starvation."""
        n_added = 0

        for prefer_non_caution in (True, False):
            for r in candidates:
                if n_added >= quota:
                    break
                if prefer_non_caution and r["_touches_caution"]:
                    continue
                if not prefer_non_caution and not r["_touches_caution"]:
                    continue

                key = (r["wild_type"], r["mutation_type"], r["position_idx"])
                if key in selected_keys:
                    continue
                if not all(position_counter[p] < max_per_position for p in r["_pos"]):
                    continue

                selected.append({**r, "category": category})
                selected_keys.add(key)
                for p in r["_pos"]:
                    position_counter[p] += 1
                n_added += 1
            if n_added >= quota:
                break

        return n_added

    # 0. Literature-verified combos: force-included, no cap, no caution logic --
    # ground truth is ground truth regardless of any score.
    n_lit = 0
    for r in rows:
        if matches_known_combo(r):
            key = (r["wild_type"], r["mutation_type"], r["position_idx"])
            if key not in selected_keys:
                selected.append({**r, "category": "literature_combo"})
                selected_keys.add(key)
                n_lit += 1
    print(f"[pair_pool] Literature-verified combos (force-included): {n_lit}")

    # 1. Top by raw FoldX score
    by_score = sorted([r for r in rows if r["_score"] is not None and r["_score"] > 0],
                       key=lambda r: r["_score"], reverse=True)
    score_position_counter = Counter()
    n_score = fill_category(by_score, n_top_score, "top_foldx_score", score_position_counter)
    print(f"[pair_pool] Top FoldX score: {n_score}/{n_top_score}")

    # 2. Top by epistasis (genuinely stabilizing candidates only), OWN position counter
    by_epi = sorted([r for r in rows if r["_epistasis"] is not None
                      and r["_score"] is not None and r["_score"] > min_combo_score],
                     key=lambda r: r["_epistasis"], reverse=True)
    epi_position_counter = Counter()
    n_epi = fill_category(by_epi, n_top_epistasis, "top_epistasis", epi_position_counter)
    print(f"[pair_pool] Top epistasis (combo_score > {min_combo_score}): {n_epi}/{n_top_epistasis}")

    fieldnames = ["wild_type", "mutation_type", "position_idx", "foldx_combo_score",
                  "foldx_epistasis", "esm_additive_score", "is_flexible_region",
                  "max_triad_embedding_drift", "drift_flag", "touches_caution", "category"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected:
            row_out = {k: r.get(k, "") for k in fieldnames if k != "touches_caution"}
            row_out["touches_caution"] = r["_touches_caution"]
            writer.writerow(row_out)

    print(f"\n[pair_pool] TOTAL: {len(selected)} -> {output_path}")
    n_caution = sum(1 for r in selected if r["_touches_caution"])
    print(f"[pair_pool] {n_caution}/{len(selected)} touch caution positions (90/168) -- "
          f"included only after non-caution quota was filled first; now recorded in output")

    all_positions = Counter()
    for r in selected:
        for p in r["_pos"]:
            all_positions[p] += 1
    print(f"\n[pair_pool] Combined position usage (top 10, across both categories):")
    for pos, count in all_positions.most_common(10):
        print(f"    {pos}: {count}x")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_table", type=str, default="results/combo_master_table.csv")
    ap.add_argument("--output", type=str, default="results/validated_pair_pool.csv")
    ap.add_argument("--n_top_score", type=int, default=40)
    ap.add_argument("--n_top_epistasis", type=int, default=25)
    ap.add_argument("--max_per_position", type=int, default=4)
    ap.add_argument("--min_combo_score", type=float, default=1.5)
    args = ap.parse_args()

    build_pool(args.master_table, args.output, n_top_score=args.n_top_score,
               n_top_epistasis=args.n_top_epistasis, max_per_position=args.max_per_position,
               min_combo_score=args.min_combo_score)