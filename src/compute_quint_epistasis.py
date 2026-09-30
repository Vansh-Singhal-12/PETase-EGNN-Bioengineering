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


def load_quad_lookup(quad_pool_csv):
    lookup = {}
    with open(quad_pool_csv) as f:
        for r in csv.DictReader(f):
            pos = tuple(sorted(int(p) for p in r["position_idx"].split(",")))
            lookup[pos] = float(r["foldx_quad_score"])
    return lookup


def load_single_lookup(singlepoint_ranked_csv):
    lookup = {}
    with open(singlepoint_ranked_csv) as f:
        for r in csv.DictReader(f):
            pos = int(r["position_idx"].split(",")[0])
            lookup[(pos, r["mutation_type"])] = -float(r["foldx_ddg_raw"])
    return lookup


def compute_quint_epistasis(quint_average_fxout_path, quint_csv_path,
                              quad_pool_csv, singlepoint_ranked_csv, output_path,
                              min_added_single_score=-2.0):
    ddg_raw = parse_average_fxout(quint_average_fxout_path)
    with open(quint_csv_path) as f:
        quint_rows = list(csv.DictReader(f))

    if len(ddg_raw) != len(quint_rows):
        print(f"[quint_epistasis] WARNING: {len(quint_rows)} rows vs {len(ddg_raw)} FoldX results -- must match 1:1.")

    quad_lookup = load_quad_lookup(quad_pool_csv)
    single_lookup = load_single_lookup(singlepoint_ranked_csv)

    results = []
    n_missing = 0
    n_filtered = 0
    n_no_parent = 0
    for row, ddg in zip(quint_rows, ddg_raw):
        quint_score = -ddg
        mut_parts = row["mutation_type"].split(";")
        pos_parts = [int(p) for p in row["position_idx"].split(",")]

        parent_quad_pos = tuple(sorted(pos_parts[:4]))
        added_pos = pos_parts[4]
        added_mut = mut_parts[4]

        parent_score = quad_lookup.get(parent_quad_pos)
        added_single_score = single_lookup.get((added_pos, added_mut))

        if parent_score is None:
            n_no_parent += 1
            additive_expectation = None
            epistasis = None
        elif added_single_score is not None and added_single_score < min_added_single_score:
            n_filtered += 1
            additive_expectation = None
            epistasis = None
        elif added_single_score is None:
            n_missing += 1
            additive_expectation = None
            epistasis = None
        else:
            additive_expectation = parent_score + added_single_score
            epistasis = quint_score - additive_expectation

        results.append({
            "wild_type": row["wild_type"],
            "mutation_type": row["mutation_type"],
            "position_idx": row["position_idx"],
            "foldx_quint_score": quint_score,
            "parent_quad_score": parent_score,
            "added_single_score": added_single_score,
            "additive_expectation": additive_expectation,
            "foldx_quint_epistasis": epistasis,
        })

    print(f"[quint_epistasis] {n_no_parent}/{len(results)} had no matching parent quad "
          f"(expected for literature reference variants, e.g. FAST-PETase)")
    if n_filtered:
        print(f"[quint_epistasis] {n_filtered}/{len(results)} excluded (added single below {min_added_single_score})")
    if n_missing:
        print(f"[quint_epistasis] {n_missing}/{len(results)} missing single lookup")

    results.sort(key=lambda r: r["foldx_quint_score"], reverse=True)

    fieldnames = ["wild_type", "mutation_type", "position_idx", "foldx_quint_score",
                  "parent_quad_score", "added_single_score", "additive_expectation",
                  "foldx_quint_epistasis"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[quint_epistasis] Wrote {len(results)} quints -> {output_path}\n")
    print("[quint_epistasis] TOP 15 by raw score:")
    for r in results[:15]:
        epi = f"{r['foldx_quint_epistasis']:.3f}" if r["foldx_quint_epistasis"] is not None else "n/a"
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | score: {r['foldx_quint_score']:.3f} | epistasis: {epi}")

    fast_petase_pos = {121, 186, 224, 233, 280}
    for r in results:
        pos_set = set(int(p) for p in r["position_idx"].split(","))
        if pos_set == fast_petase_pos:
            print(f"\n[quint_epistasis] FAST-PETase reference: score={r['foldx_quint_score']:.3f}")
            break

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quint_average_fxout", type=str, required=True)
    ap.add_argument("--quint_csv", type=str, default="data/screening_round5_quints.csv")
    ap.add_argument("--quad_pool", type=str, default="results/validated_quad_pool.csv")
    ap.add_argument("--singlepoint_ranked", type=str, default="results/foldx_singlepoint_ranked.csv")
    ap.add_argument("--output", type=str, default="results/foldx_quint_epistasis.csv")
    ap.add_argument("--min_added_single_score", type=float, default=-2.0)
    args = ap.parse_args()

    compute_quint_epistasis(args.quint_average_fxout, args.quint_csv,
                             args.quad_pool, args.singlepoint_ranked, args.output,
                             min_added_single_score=args.min_added_single_score)