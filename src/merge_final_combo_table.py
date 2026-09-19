"""
Merges foldx_combo_full.csv (FoldX score + epistasis + B-factor flexibility
+ ESM additive score) with triad_drift_scores.csv (EGNN backbone embedding
drift at the catalytic triad) into one complete consensus table for combo
selection.

Join key: (wild_type, mutation_type, position_idx) -- these should match
exactly since both files derive from the same original combo pool
(data/screening_round2_pairs_v2.csv), just processed through different
scripts. Any row that fails to match is reported.
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
        print(f"[merge_final] WARNING: {n_unmatched}/{len(merged)} rows had no matching drift score -- "
              f"check that both files derive from the same original combo pool in the same key format.")
    else:
        print(f"[merge_final] All {len(merged)} rows matched cleanly.")

    fieldnames = list(merged[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    print(f"[merge_final] Wrote {len(merged)} rows -> {output_path}")
    print(f"[merge_final] Columns: {', '.join(fieldnames)}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--foldx_full", type=str, default="results/foldx_combo_full.csv")
    ap.add_argument("--triad_drift", type=str, default="results/triad_drift_scores.csv")
    ap.add_argument("--output", type=str, default="results/combo_master_table.csv")
    args = ap.parse_args()

    merge_tables(args.foldx_full, args.triad_drift, args.output)