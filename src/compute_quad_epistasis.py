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


def load_triple_lookup(triple_pool_csv):
    """(sorted tuple of 3 positions) -> foldx_triple_score"""
    lookup = {}
    with open(triple_pool_csv) as f:
        for r in csv.DictReader(f):
            pos = tuple(sorted(int(p) for p in r["position_idx"].split(",")))
            lookup[pos] = float(r["foldx_triple_score"])
    return lookup


def load_single_lookup(singlepoint_ranked_csv):
    lookup = {}
    with open(singlepoint_ranked_csv) as f:
        for r in csv.DictReader(f):
            pos = int(r["position_idx"].split(",")[0])
            lookup[(pos, r["mutation_type"])] = -float(r["foldx_ddg_raw"])
    return lookup


def compute_quad_epistasis(quad_average_fxout_path, quad_csv_path,
                             triple_pool_csv, singlepoint_ranked_csv, output_path,
                             min_added_single_score=-2.0):
    ddg_raw = parse_average_fxout(quad_average_fxout_path)
    with open(quad_csv_path) as f:
        quad_rows = list(csv.DictReader(f))

    if len(ddg_raw) != len(quad_rows):
        print(f"[quad_epistasis] WARNING: {len(quad_rows)} rows vs {len(ddg_raw)} FoldX results -- must match 1:1.")

    triple_lookup = load_triple_lookup(triple_pool_csv)
    single_lookup = load_single_lookup(singlepoint_ranked_csv)

    results = []
    n_missing = 0
    n_filtered = 0
    for row, ddg in zip(quad_rows, ddg_raw):
        quad_score = -ddg
        mut_parts = row["mutation_type"].split(";")
        pos_parts = [int(p) for p in row["position_idx"].split(",")]

        parent_triple_pos = tuple(sorted(pos_parts[:3]))
        added_pos = pos_parts[3]
        added_mut = mut_parts[3]

        parent_score = triple_lookup.get(parent_triple_pos)
        added_single_score = single_lookup.get((added_pos, added_mut))

        if added_single_score is not None and added_single_score < min_added_single_score:
            n_filtered += 1
            additive_expectation = None
            epistasis = None
        elif parent_score is None or added_single_score is None:
            n_missing += 1
            additive_expectation = None
            epistasis = None
        else:
            additive_expectation = parent_score + added_single_score
            epistasis = quad_score - additive_expectation

        results.append({
            "wild_type": row["wild_type"],
            "mutation_type": row["mutation_type"],
            "position_idx": row["position_idx"],
            "foldx_quad_score": quad_score,
            "parent_triple_score": parent_score,
            "added_single_score": added_single_score,
            "additive_expectation": additive_expectation,
            "foldx_quad_epistasis": epistasis,
        })

    if n_missing:
        print(f"[quad_epistasis] {n_missing}/{len(results)} missing parent triple or single lookup")
    if n_filtered:
        print(f"[quad_epistasis] {n_filtered}/{len(results)} excluded (added single below {min_added_single_score})")

    results.sort(key=lambda r: r["foldx_quad_score"], reverse=True)

    fieldnames = ["wild_type", "mutation_type", "position_idx", "foldx_quad_score",
                  "parent_triple_score", "added_single_score", "additive_expectation",
                  "foldx_quad_epistasis"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[quad_epistasis] Wrote {len(results)} quads -> {output_path}\n")
    print("[quad_epistasis] TOP 15 by raw score:")
    for r in results[:15]:
        epi = f"{r['foldx_quad_epistasis']:.3f}" if r["foldx_quad_epistasis"] is not None else "n/a"
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | score: {r['foldx_quad_score']:.3f} | epistasis: {epi}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quad_average_fxout", type=str, required=True)
    ap.add_argument("--quad_csv", type=str, default="data/screening_round4_quads.csv")
    ap.add_argument("--triple_pool", type=str, default="results/validated_triple_pool.csv")
    ap.add_argument("--singlepoint_ranked", type=str, default="results/foldx_singlepoint_ranked.csv")
    ap.add_argument("--output", type=str, default="results/foldx_quad_epistasis.csv")
    ap.add_argument("--min_added_single_score", type=float, default=-2.0)
    args = ap.parse_args()

    compute_quad_epistasis(args.quad_average_fxout, args.quad_csv,
                            args.triple_pool, args.singlepoint_ranked, args.output,
                            min_added_single_score=args.min_added_single_score)