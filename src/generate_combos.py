import argparse
import csv
import itertools


def generate_pairwise_combos(pool_csv_path, output_path, max_combos=None):
    with open(pool_csv_path) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["_pos"] = int(r["position_idx"].split(",")[0])

    combos = []
    for r1, r2 in itertools.combinations(rows, 2):
        if r1["_pos"] == r2["_pos"]:
            continue  # can't mutate the same position twice in one candidate
        combos.append({
            "wild_type": f"{r1['wild_type']};{r2['wild_type']}",
            "mutation_type": f"{r1['mutation_type']};{r2['mutation_type']}",
            "position_idx": f"{r1['_pos']},{r2['_pos']}",
            "stability_score": 0.0,  # placeholder, same as Round 1 library
            "protein_id": "6EQE",
            "_component_a": f"{r1['_pos']}:{r1['mutation_type']}",
            "_component_b": f"{r2['_pos']}:{r2['mutation_type']}",
        })

    if max_combos is not None and len(combos) > max_combos:
        print(f"[generate_combos] {len(combos)} possible pairs exceeds max_combos={max_combos}; "
              f"truncating (consider a smaller pool if this matters).")
        combos = combos[:max_combos]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "stability_score", "protein_id",
                                                 "_component_a", "_component_b"])
        writer.writeheader()
        writer.writerows(combos)

    print(f"[generate_combos] Generated {len(combos)} pairwise combinations from "
          f"{len(rows)} pool candidates -> {output_path}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_csv", type=str, default="results/diversified_pool.csv")
    ap.add_argument("--output", type=str, default="data/screening_round2_pairwise.csv")
    ap.add_argument("--max_combos", type=int, default=None,
                     help="Optional cap if the pool produces too many pairs to score in reasonable time")
    args = ap.parse_args()

    generate_pairwise_combos(args.pool_csv, args.output, max_combos=args.max_combos)