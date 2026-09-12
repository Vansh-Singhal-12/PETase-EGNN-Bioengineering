"""
Parses FoldX's Average_*.fxout and merges it with foldx_shortlist.csv
(same row order as individual_list.txt, which was generated directly from
the shortlist -- see convert_shortlist_to_foldx.py's ordering note).

Sign convention: FoldX reports ddG where NEGATIVE = stabilizing, POSITIVE =
destabilizing (energy change upon mutation). This is flipped from the
EGNN's convention (positive predicted_score = more stabilizing). This
script flips FoldX's sign so both columns agree: positive = stabilizing,
in both, for direct comparison.
"""
import argparse
import csv


def parse_average_fxout(path):
    """Returns list of (pdb_name, ddg_foldx_raw) in file order.
    Finds the 'total energy' column by NAME from the header row, rather
    than assuming a fixed index -- the Average file has an 'SD' column
    between 'Pdb' and 'total energy' that a fixed-index read would miss."""
    results = []
    with open(path) as f:
        lines = f.readlines()

    header_idx = None
    col_names = None
    for i, line in enumerate(lines):
        if line.startswith("Pdb\t"):
            header_idx = i
            col_names = line.strip().split("\t")
            break
    if header_idx is None:
        raise ValueError(f"Couldn't find data header row in {path}")

    energy_col_idx = col_names.index("total energy")

    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        pdb_name = parts[0]
        total_energy = float(parts[energy_col_idx])
        results.append((pdb_name, total_energy))

    return results


def merge_and_compare(shortlist_path, average_fxout_path, output_path):
    with open(shortlist_path) as f:
        shortlist_rows = list(csv.DictReader(f))

    foldx_results = parse_average_fxout(average_fxout_path)

    if len(foldx_results) != len(shortlist_rows):
        print(f"[parse_foldx_results] WARNING: shortlist has {len(shortlist_rows)} rows but "
              f"FoldX output has {len(foldx_results)} rows -- these should match 1:1 by order. "
              f"Proceeding with min(len) but DOUBLE-CHECK this before trusting the merge.")

    merged = []
    for shortlist_row, (pdb_name, ddg_raw) in zip(shortlist_rows, foldx_results):
        ddg_foldx_flipped = -ddg_raw  # flip sign: now positive = stabilizing, matching EGNN convention
        egnn_score = float(shortlist_row.get("combo_score", 0) or 0)

        merged.append({
            "wild_type": shortlist_row["wild_type"],
            "mutation_type": shortlist_row["mutation_type"],
            "position_idx": shortlist_row["position_idx"],
            "category": shortlist_row["category"],
            "egnn_score": egnn_score,
            "foldx_ddg_raw": ddg_raw,
            "foldx_score_flipped": ddg_foldx_flipped,  # positive = stabilizing, same direction as EGNN now
            "agree_direction": (egnn_score > 0) == (ddg_foldx_flipped > 0),
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "category", "egnn_score", "foldx_ddg_raw",
                                                 "foldx_score_flipped", "agree_direction"])
        writer.writeheader()
        writer.writerows(merged)

    print(f"[parse_foldx_results] Merged {len(merged)} candidates -> {output_path}\n")

    # Category-level breakdown -- the key diagnostic
    from collections import defaultdict
    by_category = defaultdict(list)
    for r in merged:
        by_category[r["category"]].append(r)

    print("[parse_foldx_results] AGREEMENT RATE BY CATEGORY (EGNN vs FoldX direction):")
    for cat, rows in sorted(by_category.items()):
        n_agree = sum(1 for r in rows if r["agree_direction"])
        avg_foldx = sum(r["foldx_score_flipped"] for r in rows) / len(rows)
        avg_egnn = sum(r["egnn_score"] for r in rows) / len(rows)
        print(f"    {cat:<20}: {n_agree}/{len(rows)} agree ({100*n_agree/len(rows):.0f}%) | "
              f"avg EGNN: {avg_egnn:.3f} | avg FoldX (flipped): {avg_foldx:.3f}")

    print(f"\n[parse_foldx_results] BIAS-CHECK SPECIFIC RESULT:")
    bias_rows = by_category.get("bias_check", [])
    n_foldx_rejects = sum(1 for r in bias_rows if r["foldx_score_flipped"] < 0)
    print(f"    Of {len(bias_rows)} bias_check candidates, FoldX flags {n_foldx_rejects} as "
          f"DESTABILIZING despite high EGNN scores.")
    if len(bias_rows) > 0 and n_foldx_rejects / len(bias_rows) > 0.5:
        print(f"    -> This SUPPORTS the burial-bias diagnosis: FoldX disagrees with the EGNN on "
              f"the majority of the deliberately-flagged bulky->small/buried candidates.")
    elif len(bias_rows) > 0:
        print(f"    -> FoldX agrees with the EGNN on most of these -- worth re-examining whether "
              f"the bias is as strong as the burial-proxy diagnostic suggested, or whether these "
              f"specific positions genuinely tolerate the substitution.")

    top_by_egnn = sorted(merged, key=lambda r: r["egnn_score"], reverse=True)[:15]
    print(f"\n[parse_foldx_results] TOP 15 BY EGNN SCORE, WITH FOLDX CROSS-CHECK:")
    for r in top_by_egnn:
        flag = "AGREE" if r["agree_direction"] else "DISAGREE"
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} [{r['category']}] | "
              f"EGNN: {r['egnn_score']:.3f} | FoldX: {r['foldx_score_flipped']:.3f} | {flag}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", type=str, default="results/foldx_shortlist.csv")
    ap.add_argument("--average_fxout", type=str, required=True,
                     help="Path to Average_6EQE_Repair.fxout")
    ap.add_argument("--output", type=str, default="results/foldx_comparison.csv")
    args = ap.parse_args()

    merge_and_compare(args.shortlist, args.average_fxout, args.output)