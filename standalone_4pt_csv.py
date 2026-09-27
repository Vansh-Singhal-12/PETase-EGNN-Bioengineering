import csv

standalone_quad = [{
    "wild_type": "N;N;N;S", "mutation_type": "I;K;K;V", "position_idx": "114,205,233,269",
}]

with open("data/standalone_quad_1.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                             "stability_score", "protein_id"])
    writer.writeheader()
    for r in standalone_quad:
        writer.writerow({**r, "stability_score": 0.0, "protein_id": "6EQE"})
print("Saved data/standalone_quad_1.csv")