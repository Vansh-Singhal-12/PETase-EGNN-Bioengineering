"""
Selects the validated quad (4-point) pool from quad_master_table.csv.

Three groups, unioned together:
  1. literature_triple_extension -- quads built by extending the real,
     literature-verified ThermoPETase triple (S121E/D186H/R280A, Son et
     al. 2019). Ranked by FoldX score, capped -- keeps the lineage of the
     one real multi-point literature anchor traceable forward, same
     principle as the pair->triple stage.
  2. top_foldx_score -- top N purely by FoldX score.
  3. top_epistasis -- top N by epistasis, restricted to genuinely
     stabilizing candidates (score > min_combo_score).

Selection is purely score/epistasis-ranked plus the position cap.
touches_caution (positions 90/168) and drift_flag remain as REPORTED
columns for manual review at the docking/MD stage -- confirmed NOT to
gate selection, after the earlier tiered-filtering bug silently excluded
top-scoring candidates whenever the "clean" tier alone could fill a quota.
"""
import argparse
import csv
from collections import Counter

CAUTION_POSITIONS = {90, 168}
LITERATURE_TRIPLE_POSITIONS = frozenset({121, 186, 280})  # S121E/D186H/R280A, ThermoPETase, real


def build_pool(master_table_path, output_path, n_top_score=25, n_top_epistasis=15,
               max_per_position=3, min_combo_score=1.5, n_literature_cap=8):
    with open(master_table_path) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["_score"] = float(r["foldx_quad_score"]) if r.get("foldx_quad_score") else None
        r["_epistasis"] = float(r["foldx_quad_epistasis"]) if r.get("foldx_quad_epistasis") else None
        r["_pos"] = [int(p) for p in r["position_idx"].split(",")]
        r["_touches_caution"] = any(p in CAUTION_POSITIONS for p in r["_pos"])
        r["_is_drift_flagged"] = str(r.get("drift_flag", "")).strip() == "True"

    selected = []
    selected_keys = set()

    def fill_category(candidates, quota, category, position_counter):
        n_added = 0
        for r in candidates:
            if n_added >= quota:
                break
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
        return n_added

    # 1. Literature-triple-extended quads: collect all matches, sort by
    # FoldX score, take the best n_literature_cap.
    lit_candidates = []
    for r in rows:
        triple_part = tuple(sorted(r["_pos"][:3]))
        if set(triple_part) == LITERATURE_TRIPLE_POSITIONS:
            lit_candidates.append(r)

    lit_candidates.sort(key=lambda r: r["_score"] if r["_score"] is not None else -999, reverse=True)

    n_lit = 0
    for r in lit_candidates[:n_literature_cap]:
        key = (r["wild_type"], r["mutation_type"], r["position_idx"])
        if key not in selected_keys:
            selected.append({**r, "category": "literature_triple_extension"})
            selected_keys.add(key)
            n_lit += 1
    print(f"[quad_pool] Literature-triple-extended quads "
          f"(best {n_literature_cap} by FoldX score, out of {len(lit_candidates)} matches): {n_lit}")

    # 2. Top by raw FoldX score
    by_score = sorted([r for r in rows if r["_score"] is not None and r["_score"] > 0],
                       key=lambda r: r["_score"], reverse=True)
    score_counter = Counter()
    n_score = fill_category(by_score, n_top_score, "top_foldx_score", score_counter)
    print(f"[quad_pool] Top FoldX score: {n_score}/{n_top_score}")

    # 3. Top by epistasis (genuinely stabilizing only), own position counter
    by_epi = sorted([r for r in rows if r["_epistasis"] is not None
                      and r["_score"] is not None and r["_score"] > min_combo_score],
                     key=lambda r: r["_epistasis"], reverse=True)
    epi_counter = Counter()
    n_epi = fill_category(by_epi, n_top_epistasis, "top_epistasis", epi_counter)
    print(f"[quad_pool] Top epistasis (score > {min_combo_score}): {n_epi}/{n_top_epistasis}")

    fieldnames = ["wild_type", "mutation_type", "position_idx", "foldx_quad_score",
                  "foldx_quad_epistasis", "esm_additive_score", "is_flexible_region",
                  "max_triad_embedding_drift", "drift_flag", "touches_caution", "category"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected:
            row_out = {k: r.get(k, "") for k in fieldnames if k != "touches_caution"}
            row_out["touches_caution"] = r["_touches_caution"]
            writer.writerow(row_out)

    print(f"\n[quad_pool] TOTAL: {len(selected)} -> {output_path}")
    n_caution = sum(1 for r in selected if r["_touches_caution"])
    n_drift = sum(1 for r in selected if r["_is_drift_flagged"])
    print(f"[quad_pool] {n_caution}/{len(selected)} touch caution positions (90/168) -- INFORMATIONAL")
    print(f"[quad_pool] {n_drift}/{len(selected)} flagged high triad-embedding-drift -- INFORMATIONAL")

    category_counts = Counter(r["category"] for r in selected)
    print(f"\n[quad_pool] Breakdown by category:")
    for cat, count in category_counts.most_common():
        print(f"    {cat:<28}: {count}")

    all_positions = Counter()
    for r in selected:
        for p in r["_pos"]:
            all_positions[p] += 1
    print(f"\n[quad_pool] Position usage (top 10):")
    for pos, count in all_positions.most_common(10):
        print(f"    {pos}: {count}x")

    top_score_in_pool = max((r["_score"] for r in selected if r["_score"] is not None), default=None)
    print(f"\n[quad_pool] Sanity check -- highest score in final pool: {top_score_in_pool}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_table", type=str, default="results/quad_master_table.csv")
    ap.add_argument("--output", type=str, default="results/validated_quad_pool.csv")
    ap.add_argument("--n_top_score", type=int, default=25)
    ap.add_argument("--n_top_epistasis", type=int, default=15)
    ap.add_argument("--max_per_position", type=int, default=3)
    ap.add_argument("--min_combo_score", type=float, default=1.5)
    ap.add_argument("--n_literature_cap", type=int, default=8)
    args = ap.parse_args()

    build_pool(args.master_table, args.output, n_top_score=args.n_top_score,
               n_top_epistasis=args.n_top_epistasis, max_per_position=args.max_per_position,
               min_combo_score=args.min_combo_score, n_literature_cap=args.n_literature_cap)