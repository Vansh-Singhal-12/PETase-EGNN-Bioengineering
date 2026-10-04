import csv

with open("results/large_combos_master_table.csv") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r["_score"] = float(r["foldx_large_score"]) if r.get("foldx_large_score") else None

rows_sorted = sorted([r for r in rows if r["_score"] is not None], key=lambda r: r["_score"], reverse=True)

fieldnames = [k for k in rows[0].keys() if k != "_score"]
with open("results/large_combos_master_table.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows_sorted:
        writer.writerow({k: r[k] for k in fieldnames})

print(f"Sorted {len(rows_sorted)} rows by foldx_large_score (most stabilizing first)\n")
print("TOP 15:")
for r in rows_sorted[:15]:
    print(f"    [{r['position_idx']}] {r['wild_type']}->{r['mutation_type']} | "
          f"score: {r['_score']:.3f} | source: {r.get('_source_mechanism', 'n/a')} | "
          f"drift_flag: {r.get('drift_flag', '')}")