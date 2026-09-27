import csv

with open("results/standalone_quad_1_full.csv") as f:
    full_row = list(csv.DictReader(f))[0]

with open("results/standalone_quad_1_drift.csv") as f:
    drift_row = list(csv.DictReader(f))[0]

pos_list = [int(p) for p in full_row["position_idx"].split(",")]
touches_caution = any(p in {90, 168} for p in pos_list)

new_row = {
    "wild_type": full_row["wild_type"],
    "mutation_type": full_row["mutation_type"],
    "position_idx": full_row["position_idx"],
    "foldx_quad_score": full_row["foldx_triple_score"],  # reused script's column name
    "foldx_quad_epistasis": full_row.get("epistasis_3singles", ""),
    "esm_additive_score": full_row.get("esm_additive_score", ""),
    "is_flexible_region": full_row.get("is_flexible_region", ""),
    "max_triad_embedding_drift": drift_row.get("max_triad_embedding_drift", ""),
    "drift_flag": drift_row.get("drift_flag", ""),
    "touches_caution": touches_caution,
    "category": "rational_design",
}

with open("results/validated_quad_pool.csv") as f:
    existing_rows = list(csv.DictReader(f))
    fieldnames = list(existing_rows[0].keys())

key = (new_row["wild_type"], new_row["mutation_type"], new_row["position_idx"])
already_present = any((r["wild_type"], r["mutation_type"], r["position_idx"]) == key for r in existing_rows)

if already_present:
    print("Already in pool, skipping.")
else:
    combined = existing_rows + [new_row]
    with open("results/validated_quad_pool.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    print(f"Added: {new_row['wild_type']}->{new_row['mutation_type']} @ {new_row['position_idx']} "
          f"(score={new_row['foldx_quad_score']})")
    print(f"New pool total: {len(combined)}")