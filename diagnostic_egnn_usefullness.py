import csv
import random

random.seed(42)

with open("results/round1_ranked.csv") as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda r: float(r["predicted_score"]), reverse=True)
n = len(rows)

# Stratified: 20 from top decile, 20 from middle, 20 from bottom decile,
# 40 fully random -- gives real spread across the score distribution.
top = rows[:20]
middle = rows[n//2 - 10 : n//2 + 10]
bottom = rows[-20:]
random_sample = random.sample(rows, 40)

selected = {tuple(r.items()) for r in (top + middle + bottom + random_sample)}
selected = [dict(t) for t in selected]

print(f"Stratified sample: {len(selected)} candidates across full score range")

from src.protein_registry import build_registry
registry = build_registry(verbose=False)
chain_id = registry["6EQE"]["chain_id"]

lines = []
for r in selected:
    lines.append(f"{r['wild_type']}{chain_id}{r['position_idx']}{r['mutation_type']};")

with open("individual_list_stratified.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

with open("results/stratified_check_shortlist.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx", "predicted_score", "n_mutations"])
    writer.writeheader()
    writer.writerows(selected)

print(f"Wrote individual_list_stratified.txt ({len(lines)} lines) and results/stratified_check_shortlist.csv")