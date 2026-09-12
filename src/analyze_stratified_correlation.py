"""
Merges the stratified FoldX results with their EGNN predicted_score
(from stratified_check_shortlist.csv) and computes the actual correlation
between the two -- this is the test that tells us whether the EGNN carries
real rescuable signal, or whether it's not meaningfully correlated with
FoldX at all, once we look beyond just its (bias-concentrated) top picks.
"""
import csv
import argparse
from scipy.stats import spearmanr, pearsonr


def parse_average_fxout(path, skip_first_n=0):
    results = []
    with open(path) as f:
        lines = f.readlines()

    header_idx = None
    col_names = None
    for i, line in enumerate(lines):
        if line.startswith("Pdb\t"):
            header_idx = i
            col_names = line.strip().split("\t")
            break
    if header_idx is None:
        raise ValueError(f"Couldn't find data header row in {path}")

    energy_col_idx = col_names.index("total energy")

    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        results.append((parts[0], float(parts[energy_col_idx])))

    return results[skip_first_n:]


def analyze(shortlist_path, average_fxout_path, skip_first_n):
    with open(shortlist_path) as f:
        shortlist_rows = list(csv.DictReader(f))

    foldx_results = parse_average_fxout(average_fxout_path, skip_first_n=skip_first_n)

    if len(foldx_results) != len(shortlist_rows):
        print(f"[WARNING] {len(shortlist_rows)} shortlist rows vs {len(foldx_results)} FoldX rows "
              f"after skipping {skip_first_n} -- these MUST match 1:1, double check skip_first_n.")

    egnn_scores, foldx_scores = [], []
    n_agree = 0
    for shortlist_row, (pdb_name, ddg_raw) in zip(shortlist_rows, foldx_results):
        egnn = float(shortlist_row["predicted_score"])
        foldx_flipped = -ddg_raw  # match EGNN convention: positive = stabilizing
        egnn_scores.append(egnn)
        foldx_scores.append(foldx_flipped)
        if (egnn > 0) == (foldx_flipped > 0):
            n_agree += 1

    rho, p_rho = spearmanr(egnn_scores, foldx_scores)
    r, p_r = pearsonr(egnn_scores, foldx_scores)

    print(f"[analyze] n = {len(egnn_scores)}")
    print(f"[analyze] Spearman rho: {rho:.4f} (p={p_rho:.4e})")
    print(f"[analyze] Pearson r:    {r:.4f} (p={p_r:.4e})")
    print(f"[analyze] Direction agreement: {n_agree}/{len(egnn_scores)} ({100*n_agree/len(egnn_scores):.0f}%)")
    print(f"[analyze] EGNN score range: {min(egnn_scores):.3f} to {max(egnn_scores):.3f}")
    print(f"[analyze] FoldX score range (flipped): {min(foldx_scores):.3f} to {max(foldx_scores):.3f}")

    if rho > 0.3 and p_rho < 0.05:
        print(f"\n[analyze] REAL SIGNAL FOUND: EGNN correlates meaningfully with FoldX across the "
              f"full stratified range, even though top-ranked picks disagreed heavily. This suggests "
              f"the EGNN's top-score bias is a RANKING problem concentrated at the extreme top, not a "
              f"total lack of signal -- rescuable via recalibration or by not trusting the extreme top "
              f"tier blindly.")
    elif rho > 0 and p_rho < 0.05:
        print(f"\n[analyze] WEAK BUT REAL correlation. Some signal, but not strong -- the EGNN should "
              f"be treated as a coarse pre-filter only, with FoldX as the primary decision-maker.")
    else:
        print(f"\n[analyze] NO RELIABLE CORRELATION FOUND. This suggests the EGNN's predictions, as "
              f"currently trained, are not usefully correlated with FoldX-estimated stability across "
              f"the full range -- not just at the biased top tier. FoldX should become the PRIMARY "
              f"screening tool going forward, with the EGNN's role reduced (e.g., only for flagging "
              f"candidates for FoldX to check, or dropped from the ranking role entirely).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", type=str, default="results/stratified_check_shortlist.csv")
    ap.add_argument("--average_fxout", type=str, required=True)
    ap.add_argument("--skip_first_n", type=int, default=3,
                     help="Rows to skip at the start of the fxout (leftover from a previous run in the same output folder)")
    args = ap.parse_args()

    analyze(args.shortlist, args.average_fxout, args.skip_first_n)