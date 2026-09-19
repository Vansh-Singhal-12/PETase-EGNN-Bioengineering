"""
Additive ESM expectation for combos: sums each position's individual
masked-marginal ESM score (from the single-point ESM run), the same
additive-baseline logic already used for FoldX epistasis. True pairwise
ESM scoring (jointly masking both positions) is a heavier, more involved
computation -- this additive approximation is the standard simplified
approach and sufficient for a consensus signal, not a precision claim.
"""
import argparse
import csv


def load_esm_lookup(esm_ranked_csv):
    lookup = {}
    with open(esm_ranked_csv) as f:
        for r in csv.DictReader(f):
            if r.get("esm_score") in ("", "MISMATCH", None):
                continue
            pos = int(r["position_idx"].split(",")[0])
            lookup[(pos, r["mutation_type"])] = float(r["esm_score"])
    return lookup


def add_esm_scores(combo_csv_path, esm_ranked_csv, output_path):
    esm_lookup = load_esm_lookup(esm_ranked_csv)

    with open(combo_csv_path) as f:
        rows = list(csv.DictReader(f))

    n_missing = 0
    for r in rows:
        mut_parts = r["mutation_type"].split(";")
        pos_parts = [int(p) for p in r["position_idx"].split(",")]
        scores = []
        missing = False
        for pos, mut in zip(pos_parts, mut_parts):
            s = esm_lookup.get((pos, mut))
            if s is None:
                missing = True
                break
            scores.append(s)
        if missing:
            n_missing += 1
            r["esm_additive_score"] = ""
        else:
            r["esm_additive_score"] = round(sum(scores), 4)

    if n_missing:
        print(f"[esm_combo] {n_missing}/{len(rows)} combos missing an ESM score for one component")

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[esm_combo] Wrote {len(rows)} rows -> {output_path}")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo_csv", type=str, default="results/foldx_combo_bfactor.csv")
    ap.add_argument("--esm_ranked", type=str, default="results/foldx_bfactor_esm_ranked.csv")
    ap.add_argument("--output", type=str, default="results/foldx_combo_full.csv")
    args = ap.parse_args()

    add_esm_scores(args.combo_csv, args.esm_ranked, args.output)