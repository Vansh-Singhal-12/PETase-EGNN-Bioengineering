"""
Extends each validated pair into 3-point combos by adding one more single
mutation from the building-block pool -- staged construction, not blind
combinatorics. Skips any addition that shares a position with the pair
already, or with itself.
"""
import argparse
import csv


def generate_triples(pairs_csv_path, singles_pool_csv_path, output_path, max_combos=None):
    with open(pairs_csv_path) as f:
        pairs = list(csv.DictReader(f))

    with open(singles_pool_csv_path) as f:
        singles = list(csv.DictReader(f))
        for s in singles:
            s["_pos"] = int(s["position_idx"].split(",")[0])

    triples = []
    for pair in pairs:
        pair_wt = pair["wild_type"].split(";")
        pair_mut = pair["mutation_type"].split(";")
        pair_pos = [int(p) for p in pair["position_idx"].split(",")]

        for single in singles:
            if single["_pos"] in pair_pos:
                continue  # can't mutate the same position twice

            triples.append({
                "wild_type": ";".join(pair_wt + [single["wild_type"]]),
                "mutation_type": ";".join(pair_mut + [single["mutation_type"]]),
                "position_idx": ",".join(str(p) for p in pair_pos + [single["_pos"]]),
                "stability_score": 0.0,
                "protein_id": "6EQE",
                "_parent_pair_category": pair.get("category", ""),
            })

    if max_combos is not None and len(triples) > max_combos:
        print(f"[triples] {len(triples)} generated exceeds max_combos={max_combos}; truncating.")
        triples = triples[:max_combos]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "stability_score", "protein_id", "_parent_pair_category"])
        writer.writeheader()
        writer.writerows(triples)

    print(f"[triples] Generated {len(triples)} 3-point combos from {len(pairs)} validated pairs "
          f"x {len(singles)} singles -> {output_path}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=str, default="results/validated_pair_pool.csv")
    ap.add_argument("--singles_pool_csv", type=str, default="results/building_block_pool.csv")
    ap.add_argument("--output", type=str, default="data/screening_round3_triples.csv")
    ap.add_argument("--max_combos", type=int, default=None)
    args = ap.parse_args()

    generate_triples(args.pairs_csv, args.singles_pool_csv, args.output, args.max_combos)