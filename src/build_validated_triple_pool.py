import argparse
import csv
from collections import Counter

CAUTION_POSITIONS = {90, 168}
LITERATURE_PAIR_POSITIONS = frozenset({168, 188})  # I168R+S188Q, the one verified real combo


def build_pool(master_table_path, output_path, n_top_score=30, n_top_epistasis=20,
               max_per_position=3, min_combo_score=1.5):
    with open(master_table_path) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["_score"] = float(r["foldx_triple_score"]) if r.get("foldx_triple_score") else None
        r["_epistasis"] = float(r["foldx_triple_epistasis"]) if r.get("foldx_triple_epistasis") else None
        r["_pos"] = [int(p) for p in r["position_idx"].split(",")]
        r["_touches_caution"] = any(p in CAUTION_POSITIONS for p in r["_pos"])
        r["_is_drift_flagged"] = str(r.get("drift_flag", "")).strip() == "True"
        r["_risk_tier"] = int(r["_touches_caution"]) + int(r["_is_drift_flagged"])

    selected = []
    selected_keys = set()

    def fill_category(candidates, quota, category, position_counter):
        n_added = 0
        for tier in (0, 1, 2):
            for r in candidates:
                if n_added >= quota:
                    break
                if r["_risk_tier"] != tier:
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

    # 1. Literature-pair-extended triples (force-included, no cap, no score threshold)
    n_lit = 0
    for r in rows:
        pair_part = tuple(sorted(r["_pos"][:2]))
        if set(pair_part) == LITERATURE_PAIR_POSITIONS:
            key = (r["wild_type"], r["mutation_type"], r["position_idx"])
            if key not in selected_keys:
                selected.append({**r, "category": "literature_pair_extension"})
                selected_keys.add(key)
                n_lit += 1
    print(f"[triple_pool] Literature-pair-extended triples (force-included): {n_lit}")

    # 2. Top by raw FoldX score
    by_score = sorted([r for r in rows if r["_score"] is not None and r["_score"] > 0],
                       key=lambda r: r["_score"], reverse=True)
    score_counter = Counter()
    n_score = fill_category(by_score, n_top_score, "top_foldx_score", score_counter)
    print(f"[triple_pool] Top FoldX score: {n_score}/{n_top_score}")

    # 3. Top by epistasis (genuinely stabilizing only), own position counter
    by_epi = sorted([r for r in rows if r["_epistasis"] is not None
                      and r["_score"] is not None and r["_score"] > min_combo_score],
                     key=lambda r: r["_epistasis"], reverse=True)
    epi_counter = Counter()
    n_epi = fill_category(by_epi, n_top_epistasis, "top_epistasis", epi_counter)
    print(f"[triple_pool] Top epistasis (score > {min_combo_score}): {n_epi}/{n_top_epistasis}")

    fieldnames = ["wild_type", "mutation_type", "position_idx", "foldx_triple_score",
                  "foldx_triple_epistasis", "esm_additive_score", "is_flexible_region",
                  "max_triad_embedding_drift", "drift_flag", "touches_caution", "category"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected:
            row_out = {k: r.get(k, "") for k in fieldnames if k != "touches_caution"}
            row_out["touches_caution"] = r["_touches_caution"]
            writer.writerow(row_out)

    print(f"\n[triple_pool] TOTAL: {len(selected)} -> {output_path}")
    n_caution = sum(1 for r in selected if r["_touches_caution"])
    n_drift = sum(1 for r in selected if r["_is_drift_flagged"])
    n_both = sum(1 for r in selected if r["_risk_tier"] == 2)
    print(f"[triple_pool] {n_caution}/{len(selected)} touch caution positions (90/168)")
    print(f"[triple_pool] {n_drift}/{len(selected)} flagged high triad-embedding-drift")
    print(f"[triple_pool] {n_both}/{len(selected)} flagged on BOTH (only included if quota required it)")

    category_counts = Counter(r["category"] for r in selected)
    print(f"\n[triple_pool] Breakdown by category:")
    for cat, count in category_counts.most_common():
        print(f"    {cat:<28}: {count}")

    all_positions = Counter()
    for r in selected:
        for p in r["_pos"]:
            all_positions[p] += 1
    print(f"\n[triple_pool] Position usage (top 10):")
    for pos, count in all_positions.most_common(10):
        print(f"    {pos}: {count}x")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_table", type=str, default="results/triple_master_table.csv")
    ap.add_argument("--output", type=str, default="results/validated_triple_pool.csv")
    ap.add_argument("--n_top_score", type=int, default=30)
    ap.add_argument("--n_top_epistasis", type=int, default=20)
    ap.add_argument("--max_per_position", type=int, default=3)
    ap.add_argument("--min_combo_score", type=float, default=1.5)
    args = ap.parse_args()

    build_pool(args.master_table, args.output, n_top_score=args.n_top_score,
               n_top_epistasis=args.n_top_epistasis, max_per_position=args.max_per_position,
               min_combo_score=args.min_combo_score)