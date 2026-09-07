import os
import argparse
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr

from src.dataset import PETaseMutationDataset, AA_PROPERTIES_NORM
from src.model import PETaseStabilityEGNN
from src.protein_registry import build_registry

RANDOM_SEED = 42


def leave_one_out_calibrated_r2(preds_arr, targets_arr, exclude_mask_fn=None):
    """
    For each point i, fits linear calibration on a subset of the OTHER
    points, then predicts point i with that fit.

    exclude_mask_fn: optional function(i) -> boolean array of length n,
    True for points to INCLUDE in the fit for held-out point i (besides
    excluding i itself, which is always excluded). If None, all other
    points are used (standard LOO). Passing a function that additionally
    excludes any row sharing a mutated position with row i implements
    Leave-Position-Cluster-Out (LPCO) instead.

    Returns (r2, loo_preds, valid_mask) -- valid_mask is False for any
    point where fewer than 2 other points remained to fit a line on
    (can't fit slope+intercept from <2 points), so those are excluded
    from the R^2 computation rather than silently producing garbage.
    """
    n = len(preds_arr)
    loo_preds = np.full(n, np.nan)
    valid_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if exclude_mask_fn is not None:
            mask &= exclude_mask_fn(i)

        if mask.sum() < 2:
            continue  # not enough points left to fit slope+intercept

        slope, intercept = np.polyfit(preds_arr[mask], targets_arr[mask], 1)
        loo_preds[i] = slope * preds_arr[i] + intercept
        valid_mask[i] = True

    if valid_mask.sum() < 2:
        return float('nan'), loo_preds, valid_mask

    t = targets_arr[valid_mask]
    p = loo_preds[valid_mask]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    return r2, loo_preds, valid_mask


def bootstrap_ci(preds_arr, targets_arr, n_boot=1000, seed=RANDOM_SEED):
    """
    Bootstrap 95% CI for Spearman rho, Pearson r, and LOO-calibrated R^2.
    Resamples (pred, target) pairs with replacement n_boot times.

    Caveat (report this alongside the numbers): because resampling is with
    replacement, a resample can contain duplicate points, which is
    standard for bootstrap CIs but means the LOO fit within each resample
    is not a strict repeat of the single-sample LOO procedure above --
    treat this CI as an estimate of sampling variability given n=25, not
    as a second independent LOO evaluation.
    """
    rng = np.random.RandomState(seed)
    n = len(preds_arr)
    rhos, rs, r2s = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        p_boot, t_boot = preds_arr[idx], targets_arr[idx]

        if len(set(t_boot)) > 1:
            rho, _ = spearmanr(p_boot, t_boot)
            r, _ = pearsonr(p_boot, t_boot)
            rhos.append(rho)
            rs.append(r)

        r2_boot, _, valid = leave_one_out_calibrated_r2(p_boot, t_boot)
        if valid.sum() >= 2 and not np.isnan(r2_boot):
            r2s.append(r2_boot)

    def ci(vals):
        if len(vals) < 10:
            return (float('nan'), float('nan'))
        return (np.percentile(vals, 2.5), np.percentile(vals, 97.5))

    return {
        "spearman_ci": ci(rhos), "pearson_ci": ci(rs), "r2_loo_ci": ci(r2s),
        "n_boot_valid_r2": len(r2s),
    }


def property_only_baseline(dataset):
    """
    Naive baseline with NO graph structure, NO EGNN, NO learned weights at
    all: for each row, sums the raw (already Z-score normalized) property
    deltas across all mutated positions into a single scalar feature, per
    property (4 properties: volume, hydropathy, charge, H-bond capacity),
    then uses those 4 scalars directly as the "prediction" via the same
    LOO linear-calibration procedure used for the real model. This tells
    us how much of the EGNN's benchmark signal is explainable by "which
    properties changed and by how much" alone, with zero spatial/structural
    reasoning -- the comparison point item #8 calls for.
    """
    feats, targets = [], []
    for item in dataset.items:
        pos_list, score, wt_list, mut_list, protein_key, source_tag = item
        delta_sum = np.zeros(4)
        for w_code, m_code in zip(wt_list, mut_list):
            w_props = np.array(AA_PROPERTIES_NORM.get(w_code, [0.0, 0.0, 0.0, 0.0]))
            m_props = np.array(AA_PROPERTIES_NORM.get(m_code, [0.0, 0.0, 0.0, 0.0]))
            delta_sum += (m_props - w_props)
        feats.append(delta_sum)
        targets.append(score)

    feats = np.array(feats)   # [n, 4]
    targets = np.array(targets)

    # Collapse the 4 property-delta dimensions into one scalar via a
    # simple unweighted L2 norm-with-sign heuristic: sum of deltas,
    # projected onto their own first principal direction would be more
    # rigorous, but for a baseline whose entire point is "simple and
    # structure-free", a plain sum is the more honest, less-tuned choice.
    baseline_feature = feats.sum(axis=1)

    r2_loo, loo_preds, valid_mask = leave_one_out_calibrated_r2(baseline_feature, targets)
    if valid_mask.sum() > 1:
        rho, p_val = spearmanr(baseline_feature[valid_mask], targets[valid_mask])
        r, _ = pearsonr(baseline_feature[valid_mask], targets[valid_mask])
    else:
        rho, p_val, r = float('nan'), float('nan'), float('nan')

    return {"r2_loo": r2_loo, "spearman_rho": rho, "p_value": p_val, "pearson_r": r}


def run_benchmark(checkpoint_path, n_boot=1000):
    print("=" * 65)
    print(f" GROUND-TRUTH HISTORICAL BENCHMARK EVALUATION (leakage-free)")
    print(f" Checkpoint: {checkpoint_path}")
    print("=" * 65)

    benchmark_path = "data/benchmark_25.csv"
    if not os.path.exists(benchmark_path):
        print(f"[ERROR] Benchmark dataset not found at {benchmark_path}")
        return

    registry = build_registry(verbose=False)
    dataset = PETaseMutationDataset(
        csv_paths=[benchmark_path],
        augment_inverse=False, augment_combinations=False,
        registry=registry,
    )
    print(f"Loaded {len(dataset)} benchmark rows from '{benchmark_path}'")

    non_6eqe = [k for k in dataset.base_graphs if k != "6EQE"]
    if non_6eqe:
        raise RuntimeError(f"Benchmark unexpectedly references non-6EQE proteins: {non_6eqe}")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        print("        Run 'python -m src.train --phase pretrain' (and optionally")
        print("        '--phase calibrate') first to produce one.")
        return

    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_preds, all_targets, all_pos_sets = [], [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            graph_data, target_score, mutation_pos, shield_mask, protein_key, source_tag = dataset[i]
            pred, _ = model(graph_data, mutation_pos)
            all_preds.append(pred.item())
            all_targets.append(target_score.item())
            # mutation_pos is a LongTensor of node indices for this row --
            # captured per-row so LPCO can group rows by shared positions.
            all_pos_sets.append(set(mutation_pos.tolist()))

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    # 1. Raw Unscaled R^2
    ss_res = np.sum((targets_arr - preds_arr) ** 2)
    ss_tot = np.sum((targets_arr - np.mean(targets_arr)) ** 2)
    r2_raw = 1.0 - (ss_res / (ss_tot + 1e-8))

    # 2. Thermodynamic Gibbs-Helmholtz Physical Unit Conversion (kcal/mol -> °C)
    phys_preds = 2.1 * preds_arr + 2.5
    ss_res_phys = np.sum((targets_arr - phys_preds) ** 2)
    r2_physical = 1.0 - (ss_res_phys / (ss_tot + 1e-8))

    # 3. Zero-Leakage Leave-One-Out (LOO) Calibrated R^2 (standard)
    r2_loo, loo_preds, loo_valid = leave_one_out_calibrated_r2(preds_arr, targets_arr)

    # 4. Reference Full-fit Calibrated R^2 (leaky -- reference only)
    slope, intercept = np.polyfit(preds_arr, targets_arr, 1)
    calibrated_preds_leaky = slope * preds_arr + intercept
    ss_res_leaky = np.sum((targets_arr - calibrated_preds_leaky) ** 2)
    r2_calibrated_leaky = 1.0 - (ss_res_leaky / (ss_tot + 1e-8))

    spearman_rho, p_val = spearmanr(preds_arr, targets_arr)
    pearson_r, _ = pearsonr(preds_arr, targets_arr)

    # 5. NEW: MAE / RMSE, computed on the honest LOO-calibrated predictions
    # (raw model output isn't in real physical units without calibration,
    # so MAE/RMSE on raw preds wouldn't mean anything in °C or kcal/mol).
    valid = loo_valid
    mae_loo = np.mean(np.abs(targets_arr[valid] - loo_preds[valid]))
    rmse_loo = np.sqrt(np.mean((targets_arr[valid] - loo_preds[valid]) ** 2))

    # 6. NEW: Bootstrap 95% CIs
    boot = bootstrap_ci(preds_arr, targets_arr, n_boot=n_boot)

    # 7. NEW: Leave-Position-Cluster-Out (LPCO) -- excludes, for each held-out
    # row, every OTHER row that shares any mutated position with it, not
    # just the row itself. Addresses the critique that pure row-level LOO
    # can leak information when e.g. position 121 appears in both a single-
    # mutation row and a combination row containing position 121.
    def lpco_exclude_fn(i):
        others_mask = np.ones(len(all_pos_sets), dtype=bool)
        for j in range(len(all_pos_sets)):
            if j == i:
                continue
            if all_pos_sets[j] & all_pos_sets[i]:
                others_mask[j] = False
        return others_mask

    r2_lpco, lpco_preds, lpco_valid = leave_one_out_calibrated_r2(
        preds_arr, targets_arr, exclude_mask_fn=lpco_exclude_fn
    )
    n_lpco_valid = int(lpco_valid.sum())
    if n_lpco_valid > 1:
        lpco_rho, lpco_p = spearmanr(preds_arr[lpco_valid], targets_arr[lpco_valid])
    else:
        lpco_rho, lpco_p = float('nan'), float('nan')

    # 8. NEW: naive property-only baseline (no graph, no EGNN, no learned weights)
    baseline = property_only_baseline(dataset)

    print("-" * 65)
    print("BENCHMARK METRICS RESULTS:")
    print(f"  - Raw R^2 (no calibration)                : {r2_raw:.4f}")
    print(f"  - Thermodynamic Physical R^2 (kcal/mol->°C): {r2_physical:.4f}")
    print(f"  - Leave-One-Out Calibrated R^2 (HONEST)    : {r2_loo:.4f}")
    print(f"      95% bootstrap CI                       : [{boot['r2_loo_ci'][0]:.4f}, {boot['r2_loo_ci'][1]:.4f}] "
          f"(n_boot valid={boot['n_boot_valid_r2']}/{n_boot})")
    print(f"  - [reference only, leaky] Full-fit Cal. R^2: {r2_calibrated_leaky:.4f}  <- do not report this")
    print(f"  - MAE (LOO-calibrated, physical units)     : {mae_loo:.4f}")
    print(f"  - RMSE (LOO-calibrated, physical units)    : {rmse_loo:.4f}")
    print(f"  - p-value                                  : {p_val:.4e}")
    print(f"  - Spearman Correlation (rho)                : {spearman_rho:.4f}  "
          f"95% CI [{boot['spearman_ci'][0]:.4f}, {boot['spearman_ci'][1]:.4f}]")
    print(f"  - Pearson Correlation (r)                  : {pearson_r:.4f}  "
          f"95% CI [{boot['pearson_ci'][0]:.4f}, {boot['pearson_ci'][1]:.4f}]")
    print("-" * 65)
    print("LEAVE-POSITION-CLUSTER-OUT (LPCO) -- stricter than standard LOO:")
    print("  excludes ALL rows sharing any mutated position with the held-out row,")
    print("  not just the row itself, to rule out position-level leakage.")
    print(f"  - LPCO R^2 ({n_lpco_valid}/{len(dataset)} rows had a valid comparison set): {r2_lpco:.4f}")
    print(f"  - LPCO Spearman rho                        : {lpco_rho:.4f} (p={lpco_p:.4e})")
    print("-" * 65)
    print("NAIVE PROPERTY-ONLY BASELINE (no graph structure, no EGNN):")
    print(f"  - Baseline LOO R^2                         : {baseline['r2_loo']:.4f}")
    print(f"  - Baseline Spearman rho                    : {baseline['spearman_rho']:.4f} (p={baseline['p_value']:.4e})")
    print(f"  - Baseline Pearson r                       : {baseline['pearson_r']:.4f}")
    print("-" * 65)

    if r2_loo >= 0.75 and p_val < 0.05 and spearman_rho >= 0.75:
        print(" SUCCESS: Ground-Truth Benchmark PASSED all targets (honest LOO R^2)!")
    elif spearman_rho >= 0.75 and p_val < 0.05:
        print(" PARTIAL: Rank order and significance PASSED. LOO-calibrated R^2 not yet at target.")
    else:
        print(" NOTICE: Model requires further work to hit benchmark thresholds.")

    return {
        "checkpoint": checkpoint_path, "r2_raw": r2_raw, "r2_physical": r2_physical,
        "r2_loo": r2_loo, "r2_calibrated_leaky": r2_calibrated_leaky,
        "mae_loo": mae_loo, "rmse_loo": rmse_loo,
        "p_value": p_val, "spearman_rho": spearman_rho, "pearson_r": pearson_r,
        "bootstrap": boot, "r2_lpco": r2_lpco, "lpco_rho": lpco_rho, "lpco_p": lpco_p,
        "baseline": baseline,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_s2648.pt",
                     help="Path to a model checkpoint.")
    ap.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap resamples for CIs.")
    args = ap.parse_args()
    run_benchmark(args.checkpoint, n_boot=args.n_boot)