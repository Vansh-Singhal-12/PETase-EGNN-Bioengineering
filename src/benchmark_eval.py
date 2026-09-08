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
    n = len(preds_arr)
    loo_preds = np.full(n, np.nan)
    valid_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if exclude_mask_fn is not None:
            mask &= exclude_mask_fn(i)

        if mask.sum() < 2:
            continue

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

    feats = np.array(feats)
    targets = np.array(targets)
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
        return

    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_preds, all_targets, all_pos_sets, all_sigma = [], [], [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            graph_data, target_score, mutation_pos, shield_mask, protein_key, source_tag = dataset[i]
            mu, log_var, _ = model(graph_data, mutation_pos)
            all_preds.append(mu.item())
            all_sigma.append(torch.exp(0.5 * log_var).item())
            all_targets.append(target_score.item())
            all_pos_sets.append(set(mutation_pos.tolist()))

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    sigma_arr = np.array(all_sigma)

    ss_res = np.sum((targets_arr - preds_arr) ** 2)
    ss_tot = np.sum((targets_arr - np.mean(targets_arr)) ** 2)
    r2_raw = 1.0 - (ss_res / (ss_tot + 1e-8))

    phys_preds = 2.1 * preds_arr + 2.5
    ss_res_phys = np.sum((targets_arr - phys_preds) ** 2)
    r2_physical = 1.0 - (ss_res_phys / (ss_tot + 1e-8))

    r2_loo, loo_preds, loo_valid = leave_one_out_calibrated_r2(preds_arr, targets_arr)

    slope, intercept = np.polyfit(preds_arr, targets_arr, 1)
    calibrated_preds_leaky = slope * preds_arr + intercept
    ss_res_leaky = np.sum((targets_arr - calibrated_preds_leaky) ** 2)
    r2_calibrated_leaky = 1.0 - (ss_res_leaky / (ss_tot + 1e-8))

    spearman_rho, p_val = spearmanr(preds_arr, targets_arr)
    pearson_r, _ = pearsonr(preds_arr, targets_arr)

    valid = loo_valid
    mae_loo = np.mean(np.abs(targets_arr[valid] - loo_preds[valid]))
    rmse_loo = np.sqrt(np.mean((targets_arr[valid] - loo_preds[valid]) ** 2))

    boot = bootstrap_ci(preds_arr, targets_arr, n_boot=n_boot)

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
    print(f"  - Mean predicted sigma (raw units)         : {sigma_arr.mean():.4f} "
          f"(range {sigma_arr.min():.4f}-{sigma_arr.max():.4f}) [diagnostic only]")
    print("-" * 65)
    print("LEAVE-POSITION-CLUSTER-OUT (LPCO):")
    print(f"  - LPCO R^2 ({n_lpco_valid}/{len(dataset)} rows had a valid comparison set): {r2_lpco:.4f}")
    print(f"  - LPCO Spearman rho                        : {lpco_rho:.4f} (p={lpco_p:.4e})")
    print("-" * 65)
    print("NAIVE PROPERTY-ONLY BASELINE:")
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
        "baseline": baseline, "mean_sigma": float(sigma_arr.mean()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_s2648.pt")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    run_benchmark(args.checkpoint, n_boot=args.n_boot)