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


def load_single_lookup(singlepoint_ranked_csv):
    lookup = {}
    with open(singlepoint_ranked_csv) as f:
        for r in csv.DictReader(f):
            pos = int(r["position_idx"].split(",")[0])
            lookup[(pos, r["mutation_type"])] = -float(r["foldx_ddg_raw"])
    return lookup


def compute_epistasis(triple_average_fxout_path, triple_csv_path,
                       singlepoint_ranked_csv, output_path):
    ddg_raw = parse_average_fxout(triple_average_fxout_path)
    with open(triple_csv_path) as f:
        rows = list(csv.DictReader(f))

    if len(ddg_raw) != len(rows):
        print(f"[rational_epistasis] WARNING: {len(rows)} rows vs {len(ddg_raw)} FoldX results -- must match 1:1.")

    single_lookup = load_single_lookup(singlepoint_ranked_csv)

    results = []
    for row, ddg in zip(rows, ddg_raw):
        triple_score = -ddg
        wt_parts = row["wild_type"].split(";")
        mut_parts = row["mutation_type"].split(";")
        pos_parts = [int(p) for p in row["position_idx"].split(",")]

        single_scores = []
        missing_positions = []
        for pos, mut in zip(pos_parts, mut_parts):
            s = single_lookup.get((pos, mut))
            if s is None:
                missing_positions.append(f"{pos}{mut}")
            else:
                single_scores.append(s)

        if missing_positions:
            print(f"[rational_epistasis] WARNING: no single-point FoldX score found for "
                  f"{', '.join(missing_positions)} (row: {row['wild_type']}->{row['mutation_type']} "
                  f"@ {row['position_idx']}) -- these positions may not have been in the original "
                  f"4978-candidate single-point library. Epistasis left blank for this row.")
            additive_expectation = None
            epistasis = None
        else:
            additive_expectation = sum(single_scores)
            epistasis = triple_score - additive_expectation

        results.append({
            "wild_type": row["wild_type"],
            "mutation_type": row["mutation_type"],
            "position_idx": row["position_idx"],
            "foldx_triple_score": triple_score,
            "additive_expectation_3singles": additive_expectation,
            "epistasis_3singles": epistasis,
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "foldx_triple_score", "additive_expectation_3singles",
                                                 "epistasis_3singles"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[rational_epistasis] Wrote {len(results)} candidates -> {output_path}\n")
    for r in results:
        epi = f"{r['epistasis_3singles']:.3f}" if r["epistasis_3singles"] is not None else "n/a (missing single-point data)"
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"score: {r['foldx_triple_score']:.3f} | epistasis (vs 3 independent singles): {epi}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--triple_average_fxout", type=str, required=True)
    ap.add_argument("--triple_csv", type=str, default="data/rational_triples.csv")
    ap.add_argument("--singlepoint_ranked", type=str, default="results/foldx_singlepoint_ranked.csv")
    ap.add_argument("--output", type=str, default="results/rational_triples_epistasis.csv")
    args = ap.parse_args()

    compute_epistasis(args.triple_average_fxout, args.triple_csv,
                       args.singlepoint_ranked, args.output)