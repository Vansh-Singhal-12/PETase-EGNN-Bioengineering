"""
Merges foldx_triple_full.csv (FoldX score + epistasis + B-factor + ESM)
with triad_drift_triples.csv (EGNN embedding drift) into one complete
consensus table for triples -- same pattern as merge_final_combo_table.py.
"""
import argparse
import csv


def merge_tables(foldx_full_path, triad_drift_path, output_path):
    with open(foldx_full_path) as f:
        foldx_rows = list(csv.DictReader(f))

    drift_lookup = {}
    with open(triad_drift_path) as f:
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
            new_row["mean_triad_embedding_drift"] = ""
            new_row["drift_flag"] = ""
        else:
            new_row["max_triad_embedding_drift"] = drift_row["max_triad_embedding_drift"]
            new_row["mean_triad_embedding_drift"] = drift_row["mean_triad_embedding_drift"]
            new_row["drift_flag"] = drift_row["drift_flag"]

        merged.append(new_row)

    if n_unmatched:
        print(f"[merge_triples] WARNING: {n_unmatched}/{len(merged)} rows had no matching drift score.")
    else:
        print(f"[merge_triples] All {len(merged)} rows matched cleanly.")

    fieldnames = list(merged[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    print(f"[merge_triples] Wrote {len(merged)} rows -> {output_path}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--foldx_full", type=str, default="results/foldx_triple_full.csv")
    ap.add_argument("--triad_drift", type=str, default="results/triad_drift_triples.csv")
    ap.add_argument("--output", type=str, default="results/triple_master_table.csv")
    args = ap.parse_args()

    merge_tables(args.foldx_full, args.triad_drift, args.output)