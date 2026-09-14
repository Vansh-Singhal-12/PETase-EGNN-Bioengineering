"""
Parses the full 4978-candidate Average_*.fxout from the single-point FoldX
run and merges it back with screening_round1_singlepoint.csv (same row
order, since individual_list_full.txt was generated directly from that
file in order).
"""
import argparse
import csv


def parse_average_fxout(path):
    with open(path) as f:
        lines = f.readlines()

    header_idx, col_names = None, None
    for i, line in enumerate(lines):
        if line.startswith("Pdb\t"):
            header_idx = i
            col_names = line.strip().split("\t")
            break
    if header_idx is None:
        raise ValueError(f"Couldn't find data header row in {path}")

    energy_col_idx = col_names.index("total energy")

    results = []
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        results.append((parts[0], float(parts[energy_col_idx])))
    return results


def merge_and_rank(library_csv_path, average_fxout_path, output_path):
    with open(library_csv_path) as f:
        library_rows = list(csv.DictReader(f))

    foldx_results = parse_average_fxout(average_fxout_path)

    if len(foldx_results) != len(library_rows):
        print(f"[WARNING] {len(library_rows)} library rows vs {len(foldx_results)} FoldX rows -- "
              f"these should match 1:1. Check for leftover/appended rows before trusting this merge.")

    merged = []
    for lib_row, (pdb_name, ddg_raw) in zip(library_rows, foldx_results):
        merged.append({
            "wild_type": lib_row["wild_type"],
            "mutation_type": lib_row["mutation_type"],
            "position_idx": lib_row["position_idx"],
            "foldx_ddg_raw": ddg_raw,
            "foldx_score_flipped": -ddg_raw,  # positive = stabilizing
        })

    merged.sort(key=lambda r: r["foldx_score_flipped"], reverse=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "foldx_ddg_raw", "foldx_score_flipped"])
        writer.writeheader()
        writer.writerows(merged)

    n_stabilizing = sum(1 for r in merged if r["foldx_score_flipped"] > 0)
    print(f"[parse_foldx_full] Merged {len(merged)} candidates -> {output_path}")
    print(f"[parse_foldx_full] {n_stabilizing}/{len(merged)} ({100*n_stabilizing/len(merged):.1f}%) "
          f"predicted stabilizing (negative FoldX ddG)")
    print(f"\n[parse_foldx_full] TOP 20 most stabilizing:")
    for r in merged[:20]:
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"FoldX ddG: {r['foldx_ddg_raw']:.3f} (flipped: {r['foldx_score_flipped']:.3f})")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=str, default="data/screening_round1_singlepoint.csv")
    ap.add_argument("--average_fxout", type=str, required=True)
    ap.add_argument("--output", type=str, default="results/foldx_singlepoint_ranked.csv")
    args = ap.parse_args()

    merge_and_rank(args.library, args.average_fxout, args.output)