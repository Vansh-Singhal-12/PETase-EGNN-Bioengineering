import argparse
import csv


def load_single_lookup(singlepoint_ranked_csv):
    """Returns dict: (position, mutation_type) -> foldx_score_flipped (positive=stabilizing)"""
    lookup = {}
    with open(singlepoint_ranked_csv) as f:
        for r in csv.DictReader(f):
            pos = int(r["position_idx"].split(",")[0])
            lookup[(pos, r["mutation_type"])] = -float(r["foldx_ddg_raw"])
    return lookup


def compute_epistasis(combo_average_fxout_path, combo_pairs_csv_path,
                       singlepoint_ranked_csv, output_path, min_combo_score=None):
    with open(combo_average_fxout_path) as f:
        lines = f.readlines()
    header_idx, col_names = None, None
    for i, line in enumerate(lines):
        if line.startswith("Pdb\t"):
            header_idx = i
            col_names = line.strip().split("\t")
            break
    if header_idx is None:
        raise ValueError(f"Couldn't find data header row in {combo_average_fxout_path}")
    energy_col_idx = col_names.index("total energy")

    combo_ddg_raw = []
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        combo_ddg_raw.append(float(parts[energy_col_idx]))

    with open(combo_pairs_csv_path) as f:
        combo_rows = list(csv.DictReader(f))

    if len(combo_ddg_raw) != len(combo_rows):
        print(f"[epistasis] WARNING: {len(combo_rows)} combo rows vs {len(combo_ddg_raw)} FoldX results -- "
              f"these must match 1:1 by order. Double-check before trusting this merge.")

    single_lookup = load_single_lookup(singlepoint_ranked_csv)

    results = []
    n_missing_single = 0
    for combo_row, ddg_raw in zip(combo_rows, combo_ddg_raw):
        combo_flipped = -ddg_raw

        wt_parts = combo_row["wild_type"].split(";")
        mut_parts = combo_row["mutation_type"].split(";")
        pos_parts = [int(p) for p in combo_row["position_idx"].split(",")]

        single_scores = []
        missing = False
        for pos, mut in zip(pos_parts, mut_parts):
            s = single_lookup.get((pos, mut))
            if s is None:
                missing = True
                break
            single_scores.append(s)

        if missing:
            n_missing_single += 1
            additive_expectation = None
            epistasis = None
        else:
            additive_expectation = sum(single_scores)
            epistasis = combo_flipped - additive_expectation

        results.append({
            "wild_type": combo_row["wild_type"],
            "mutation_type": combo_row["mutation_type"],
            "position_idx": combo_row["position_idx"],
            "foldx_combo_score": combo_flipped,
            "additive_expectation": additive_expectation,
            "foldx_epistasis": epistasis,
        })

    if n_missing_single:
        print(f"[epistasis] WARNING: {n_missing_single} combos had a component single not found in "
              f"{singlepoint_ranked_csv} -- epistasis left blank for those (additive expectation unavailable).")

    results.sort(key=lambda r: r["foldx_combo_score"], reverse=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "foldx_combo_score", "additive_expectation", "foldx_epistasis"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[epistasis] Wrote {len(results)} combos (unfiltered, full set) -> {output_path}\n")

    by_score = sorted([r for r in results if r["foldx_combo_score"] is not None],
                       key=lambda r: r["foldx_combo_score"], reverse=True)
    print("[epistasis] TOP 15 by raw FoldX combo score (most stabilizing overall, unfiltered):")
    for r in by_score[:15]:
        epi_str = f"{r['foldx_epistasis']:.3f}" if r["foldx_epistasis"] is not None else "n/a"
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"combo: {r['foldx_combo_score']:.3f} | epistasis: {epi_str}")

    # FILTERED epistasis view: only consider candidates that are genuinely
    # stabilizing on their own (foldx_combo_score > min_combo_score), not
    # just candidates whose additive expectation happens to be deeply
    # penalized (often an artifact of a miscalibrated single-point score,
    # e.g. positions like 168 where FoldX disagrees sharply with real
    # literature data -- see I168R). This prevents mistaking "less bad
    # than an inflated bad baseline" for genuine synergy.
    if min_combo_score is None:
        min_combo_score = 0.0

    filtered_for_epistasis = [r for r in results
                               if r["foldx_epistasis"] is not None
                               and r["foldx_combo_score"] is not None
                               and r["foldx_combo_score"] > min_combo_score]
    by_epistasis = sorted(filtered_for_epistasis, key=lambda r: r["foldx_epistasis"], reverse=True)

    print(f"\n[epistasis] TOP 15 by FoldX EPISTASIS (FILTERED: combo_score > {min_combo_score}, "
          f"genuinely stabilizing candidates only -- {len(filtered_for_epistasis)}/{len(results)} qualify):")
    if not by_epistasis:
        print("    (none qualify -- consider lowering --min_combo_score)")
    for r in by_epistasis[:15]:
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"combo: {r['foldx_combo_score']:.3f} | additive expectation: {r['additive_expectation']:.3f} | "
              f"epistasis: {r['foldx_epistasis']:.3f}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo_average_fxout", type=str, required=True)
    ap.add_argument("--combo_pairs_csv", type=str, default="data/screening_round2_pairs_v2.csv")
    ap.add_argument("--singlepoint_ranked", type=str, default="results/foldx_singlepoint_ranked.csv")
    ap.add_argument("--output", type=str, default="results/foldx_combo_epistasis.csv")
    ap.add_argument("--min_combo_score", type=float, default=0.0,
                     help="Only rank candidates for epistasis if their combo_score exceeds this "
                          "(default 0.0 = must be genuinely stabilizing, not just less-destabilizing "
                          "than an inflated additive baseline).")
    args = ap.parse_args()

    compute_epistasis(args.combo_average_fxout, args.combo_pairs_csv,
                       args.singlepoint_ranked, args.output,
                       min_combo_score=args.min_combo_score)