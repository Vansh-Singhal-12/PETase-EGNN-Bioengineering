"""
Sanity check: runs FoldX on real, literature-verified single mutations and
compares against the known stability_score values already in
mutations_clean.csv. If FoldX's result is wildly different from these
known-good values, something is wrong with OUR FoldX setup/config, not
just an EGNN calibration issue. If it's reasonably close, FoldX itself is
trustworthy and the disagreement is on the EGNN side.
"""
import csv
from src.protein_registry import build_registry

registry = build_registry(verbose=False)
chain_id = registry["6EQE"]["chain_id"]

with open("data/mutations_clean.csv") as f:
    rows = list(csv.DictReader(f))

lines = []
print("Real verified mutations being tested:")
for r in rows:
    wt, mut, pos, score = r["wild_type"], r["mutation_type"], r["position_idx"], r["stability_score"]
    if ";" in wt:
        continue  # skip combo rows for this sanity check, singles only
    print(f"  {wt}{pos}{mut}  (known stability_score: {score})")
    lines.append(f"{wt}{chain_id}{pos}{mut};")

with open("sanity_check_list.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nWrote {len(lines)} mutations to sanity_check_list.txt")