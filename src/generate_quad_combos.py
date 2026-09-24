import argparse
import csv


def generate_quads(triples_csv_path, singles_pool_csv_path, output_path):
    with open(triples_csv_path) as f:
        triples = list(csv.DictReader(f))

    with open(singles_pool_csv_path) as f:
        singles = list(csv.DictReader(f))
        for s in singles:
            s["_pos"] = int(s["position_idx"].split(",")[0])

    quads = []
    for triple in triples:
        triple_wt = triple["wild_type"].split(";")
        triple_mut = triple["mutation_type"].split(";")
        triple_pos = [int(p) for p in triple["position_idx"].split(",")]

        for single in singles:
            if single["_pos"] in triple_pos:
                continue

            quads.append({
                "wild_type": ";".join(triple_wt + [single["wild_type"]]),
                "mutation_type": ";".join(triple_mut + [single["mutation_type"]]),
                "position_idx": ",".join(str(p) for p in triple_pos + [single["_pos"]]),
                "stability_score": 0.0,
                "protein_id": "6EQE",
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "stability_score", "protein_id"])
        writer.writeheader()
        writer.writerows(quads)

    print(f"[quads] Generated {len(quads)} 4-point combos from {len(triples)} validated triples "
          f"x {len(singles)} singles -> {output_path}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples_csv", type=str, default="results/validated_triple_pool.csv")
    ap.add_argument("--singles_pool_csv", type=str, default="results/building_block_pool.csv")
    ap.add_argument("--output", type=str, default="data/screening_round4_quads.csv")
    args = ap.parse_args()

    generate_quads(args.triples_csv, args.singles_pool_csv, args.output)