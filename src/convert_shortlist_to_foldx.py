"""
Converts foldx_shortlist.csv into FoldX's individual_list.txt mutation
format. Each row of the shortlist becomes one line: WT+Chain+Position+Mut,
comma-separated for multi-point rows, semicolon-terminated.

FoldX requires the CHAIN ID as part of each mutation code (not just
position) -- pulled from protein_registry so it stays consistent with the
rest of the project rather than hardcoding "A" here.
"""
import argparse
import csv

from src.protein_registry import build_registry


def convert(shortlist_path, output_path, protein_key="6EQE"):
    registry = build_registry(verbose=False)
    chain_id = registry[protein_key]["chain_id"]

    with open(shortlist_path) as f:
        rows = list(csv.DictReader(f))

    lines = []
    for r in rows:
        wt_parts = r["wild_type"].split(";")
        mut_parts = r["mutation_type"].split(";")
        pos_parts = r["position_idx"].split(",")

        mutation_codes = []
        for wt, mut, pos in zip(wt_parts, mut_parts, pos_parts):
            mutation_codes.append(f"{wt}{chain_id}{pos}{mut}")

        lines.append(",".join(mutation_codes) + ";")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[convert_shortlist_to_foldx] Converted {len(lines)} candidates -> {output_path}")
    print(f"[convert_shortlist_to_foldx] First 5 lines:")
    for line in lines[:5]:
        print(f"    {line}")
    print(f"\n[convert_shortlist_to_foldx] IMPORTANT: line order in this file matches row order")
    print(f"  in {shortlist_path} -- FoldX's output will be numbered by this same order")
    print(f"  (result 1 = shortlist row 1, etc.). Don't re-sort the shortlist CSV after this")
    print(f"  step without regenerating individual_list.txt, or the mapping breaks.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", type=str, default="results/foldx_shortlist.csv")
    ap.add_argument("--output", type=str, default="individual_list.txt")
    ap.add_argument("--protein_key", type=str, default="6EQE")
    args = ap.parse_args()

    convert(args.shortlist, args.output, protein_key=args.protein_key)