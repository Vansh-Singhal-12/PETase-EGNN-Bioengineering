import csv

rational_triples = [
    {"wild_type": "T;S;S", "mutation_type": "P;N;P", "position_idx": "116,238,290"},
    {"wild_type": "S;D;T", "mutation_type": "E;H;P", "position_idx": "121,186,116"},
    {"wild_type": "S;D;W", "mutation_type": "E;H;H", "position_idx": "121,186,159"},
    {"wild_type": "S;D;S", "mutation_type": "E;H;Y", "position_idx": "121,186,238"},
    {"wild_type": "S;D;R", "mutation_type": "E;H;A", "position_idx": "121,186,280"},  # ThermoPETase, real
]

with open("data/rational_triples.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                             "stability_score", "protein_id"])
    writer.writeheader()
    for r in rational_triples:
        writer.writerow({**r, "stability_score": 0.0, "protein_id": "6EQE"})

print("Saved data/rational_triples.csv")