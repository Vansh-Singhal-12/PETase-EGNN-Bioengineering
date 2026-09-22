"""
Computes epistasis for staged 3-point combos: each candidate is a validated
pair + one added single, so the correct additive baseline is
parent_pair_foldx_score + added_single_foldx_score -- this tests whether
adding the third mutation to an already-validated pair helps, hurts, or
does nothing, which is the actual question at this stage (not "how does
the triple compare to three independent singles").

FILTER: excludes the added single from the epistasis baseline if its own
FoldX score is below min_added_single_score. Without this, cases like
I168R (a real, literature-verified stabilizing mutation that FoldX badly
miscalibrates as severely destabilizing, score ~-14.76) inflate epistasis
values artificially -- the same artifact pattern already caught and fixed
at the pairwise-combo stage, recurring here because that earlier filter
only applied to the overall combo score, not specifically to the third
mutation being added on top of a validated pair.
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
        results.append(float(parts[energy_col_idx]))
    return results


def load_pair_lookup(pair_pool_csv):
    """(sorted tuple of positions) -> foldx_combo_score, for the 2 positions in each validated pair."""
    lookup = {}
    with open(pair_pool_csv) as f:
        for r in csv.DictReader(f):
            pos = tuple(sorted(int(p) for p in r["position_idx"].split(",")))
            lookup[pos] = float(r["foldx_combo_score"])
    return lookup


def load_single_lookup(singlepoint_ranked_csv):
    lookup = {}
    with open(singlepoint_ranked_csv) as f:
        for r in csv.DictReader(f):
            pos = int(r["position_idx"].split(",")[0])
            lookup[(pos, r["mutation_type"])] = -float(r["foldx_ddg_raw"])
    return lookup


def compute_triple_epistasis(triple_average_fxout_path, triple_csv_path,
                               pair_pool_csv, singlepoint_ranked_csv, output_path,
                               min_added_single_score=-2.0):
    ddg_raw = parse_average_fxout(triple_average_fxout_path)
    with open(triple_csv_path) as f:
        triple_rows = list(csv.DictReader(f))

    if len(ddg_raw) != len(triple_rows):
        print(f"[triple_epistasis] WARNING: {len(triple_rows)} triple rows vs {len(ddg_raw)} FoldX "
              f"results -- must match 1:1. Verify row count before trusting this merge.")

    pair_lookup = load_pair_lookup(pair_pool_csv)
    single_lookup = load_single_lookup(singlepoint_ranked_csv)

    results = []
    n_missing = 0
    n_filtered_unreliable_single = 0
    for row, ddg in zip(triple_rows, ddg_raw):
        triple_score = -ddg
        wt_parts = row["wild_type"].split(";")
        mut_parts = row["mutation_type"].split(";")
        pos_parts = [int(p) for p in row["position_idx"].split(",")]

        # First 2 positions = the parent pair, 3rd = the added single
        # (matches generate_triple_combos.py's construction order)
        parent_pair_pos = tuple(sorted(pos_parts[:2]))
        added_pos = pos_parts[2]
        added_mut = mut_parts[2]

        parent_score = pair_lookup.get(parent_pair_pos)
        added_single_score = single_lookup.get((added_pos, added_mut))

        if added_single_score is not None and added_single_score < min_added_single_score:
            n_filtered_unreliable_single += 1
            additive_expectation = None
            epistasis = None
        elif parent_score is None or added_single_score is None:
            n_missing += 1
            additive_expectation = None
            epistasis = None
        else:
            additive_expectation = parent_score + added_single_score
            epistasis = triple_score - additive_expectation

        results.append({
            "wild_type": row["wild_type"],
            "mutation_type": row["mutation_type"],
            "position_idx": row["position_idx"],
            "foldx_triple_score": triple_score,
            "parent_pair_score": parent_score,
            "added_single_score": added_single_score,
            "additive_expectation": additive_expectation,
            "foldx_triple_epistasis": epistasis,
        })

    if n_missing:
        print(f"[triple_epistasis] {n_missing}/{len(results)} rows missing parent pair or single lookup")
    if n_filtered_unreliable_single:
        print(f"[triple_epistasis] {n_filtered_unreliable_single}/{len(results)} rows had an added single "
              f"with FoldX score < {min_added_single_score} -- excluded from epistasis baseline as "
              f"likely-unreliable (e.g. I168R-type miscalibration)")

    results.sort(key=lambda r: r["foldx_triple_score"], reverse=True)

    fieldnames = ["wild_type", "mutation_type", "position_idx", "foldx_triple_score",
                  "parent_pair_score", "added_single_score", "additive_expectation",
                  "foldx_triple_epistasis"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[triple_epistasis] Wrote {len(results)} triples -> {output_path}\n")

    print("[triple_epistasis] TOP 15 by raw triple score (unfiltered):")
    for r in results[:15]:
        epi = f"{r['foldx_triple_epistasis']:.3f}" if r["foldx_triple_epistasis"] is not None else "n/a"
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"score: {r['foldx_triple_score']:.3f} | epistasis vs parent+single: {epi}")

    stabilizing_only = [r for r in results if r["foldx_triple_epistasis"] is not None
                         and r["foldx_triple_score"] > 0]
    by_epi = sorted(stabilizing_only, key=lambda r: r["foldx_triple_epistasis"], reverse=True)
    print(f"\n[triple_epistasis] TOP 15 by epistasis (FILTERED: genuinely stabilizing triple AND "
          f"reliable added single, {len(stabilizing_only)} qualify):")
    if not by_epi:
        print("    (none qualify -- consider lowering --min_added_single_score or checking min_combo_score logic)")
    for r in by_epi[:15]:
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"score: {r['foldx_triple_score']:.3f} | vs parent({r['parent_pair_score']:.3f})+"
              f"single({r['added_single_score']:.3f}) | epistasis: {r['foldx_triple_epistasis']:.3f}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--triple_average_fxout", type=str, required=True)
    ap.add_argument("--triple_csv", type=str, default="data/screening_round3_triples.csv")
    ap.add_argument("--pair_pool", type=str, default="results/validated_pair_pool.csv")
    ap.add_argument("--singlepoint_ranked", type=str, default="results/foldx_singlepoint_ranked.csv")
    ap.add_argument("--output", type=str, default="results/foldx_triple_epistasis.csv")
    ap.add_argument("--min_added_single_score", type=float, default=-2.0,
                     help="Exclude the added single from the epistasis baseline if its own FoldX "
                          "score is below this (filters out cases like I168R, where FoldX's own "
                          "single-point miscalibration would inflate epistasis artificially).")
    args = ap.parse_args()

    compute_triple_epistasis(args.triple_average_fxout, args.triple_csv,
                              args.pair_pool, args.singlepoint_ranked, args.output,
                              min_added_single_score=args.min_added_single_score)