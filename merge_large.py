import csv

with open("results/large_combos_full.csv") as f:
    foldx_rows = list(csv.DictReader(f))

drift_lookup = {}
with open("results/large_combos_drift.csv") as f:
    for r in csv.DictReader(f):
        key = (r["wild_type"], r["mutation_type"], r["position_idx"])
        drift_lookup[key] = r

merged = []
n_unmatched = 0
for r in foldx_rows:
    key = (r["wild_type"], r["mutation_type"], r["position_idx"])
    drift_row = drift_lookup.get(key)
    new_row = dict(r)
    if drift_row is None:
        n_unmatched += 1
        new_row["max_triad_embedding_drift"] = ""
        new_row["drift_flag"] = ""
    else:
        new_row["max_triad_embedding_drift"] = drift_row["max_triad_embedding_drift"]
        new_row["drift_flag"] = drift_row["drift_flag"]
    merged.append(new_row)

if n_unmatched:
    print(f"WARNING: {n_unmatched}/{len(merged)} rows unmatched")
else:
    print(f"All {len(merged)} rows matched cleanly")

fieldnames = list(merged[0].keys())
with open("results/large_combos_master_table.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged)

print(f"Wrote {len(merged)} rows -> results/large_combos_master_table.csv")