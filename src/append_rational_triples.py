import argparse
import csv

CAUTION_POSITIONS = {90, 168}

# wild_type;mutation_type;position_idx (as it appears in the CSV) -> category
CATEGORY_OVERRIDE = {
    "S;D;R|E;H;A|121,186,280": "literature_combo",  # ThermoPETase, real (Son et al. 2019)
}


def append_rational(rational_full_csv, rational_drift_csv, pool_csv, output_path):
    drift_lookup = {}
    with open(rational_drift_csv) as f:
        for r in csv.DictReader(f):
            key = (r["wild_type"], r["mutation_type"], r["position_idx"])
            drift_lookup[key] = r

    with open(rational_full_csv) as f:
        rational_rows = list(csv.DictReader(f))

    new_rows = []
    for r in rational_rows:
        key = (r["wild_type"], r["mutation_type"], r["position_idx"])
        drift_row = drift_lookup.get(key, {})
        pos_parts = [int(p) for p in r["position_idx"].split(",")]
        touches_caution = any(p in CAUTION_POSITIONS for p in pos_parts)

        override_key = f"{r['wild_type']}|{r['mutation_type']}|{r['position_idx']}"
        category = CATEGORY_OVERRIDE.get(override_key, "rational_design")

        new_rows.append({
            "wild_type": r["wild_type"],
            "mutation_type": r["mutation_type"],
            "position_idx": r["position_idx"],
            "foldx_triple_score": r["foldx_triple_score"],
            "foldx_triple_epistasis": r.get("epistasis_3singles", ""),  # renamed
            "esm_additive_score": r.get("esm_additive_score", ""),
            "is_flexible_region": r.get("is_flexible_region", ""),
            "max_triad_embedding_drift": drift_row.get("max_triad_embedding_drift", ""),
            "drift_flag": drift_row.get("drift_flag", ""),
            "touches_caution": touches_caution,
            "category": category,
        })

    with open(pool_csv) as f:
        existing_rows = list(csv.DictReader(f))
        fieldnames = list(existing_rows[0].keys())

    combined = existing_rows + new_rows

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)

    print(f"[append_rational] Appended {len(new_rows)} rational-design candidates to the pool")
    for r in new_rows:
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"score: {r['foldx_triple_score']} | category: {r['category']}")
    print(f"\n[append_rational] New pool total: {len(combined)} -> {output_path}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rational_full", type=str, default="results/rational_triples_full.csv")
    ap.add_argument("--rational_drift", type=str, default="results/rational_triples_drift.csv")
    ap.add_argument("--pool_csv", type=str, default="results/validated_triple_pool.csv")
    ap.add_argument("--output", type=str, default="results/validated_triple_pool.csv")
    args = ap.parse_args()

    append_rational(args.rational_full, args.rational_drift, args.pool_csv, args.output)