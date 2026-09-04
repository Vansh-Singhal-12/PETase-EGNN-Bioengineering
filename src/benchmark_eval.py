import os
import argparse
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr

from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN
from src.protein_registry import build_registry


def leave_one_out_calibrated_r2(preds_arr, targets_arr):
    """
    For each point, fits the linear calibration on the other 24 points only,
    then predicts the held-out point with that fit. R^2 is computed over
    all 25 held-out predictions. Still correct and leakage-free regardless of
    which checkpoint is being evaluated.
    """
    n = len(preds_arr)
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        slope, intercept = np.polyfit(preds_arr[mask], targets_arr[mask], 1)
        loo_preds[i] = slope * preds_arr[i] + intercept

    ss_res = np.sum((targets_arr - loo_preds) ** 2)
    ss_tot = np.sum((targets_arr - np.mean(targets_arr)) ** 2)
    r2_loo = 1.0 - (ss_res / (ss_tot + 1e-8))
    return r2_loo, loo_preds


def run_benchmark(checkpoint_path):
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

    all_preds, all_targets = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            graph_data, target_score, mutation_pos, shield_mask, protein_key, source_tag = dataset[i]
            pred, _ = model(graph_data, mutation_pos)
            all_preds.append(pred.item())
            all_targets.append(target_score.item())

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    # 1. Raw Unscaled R^2
    ss_res = np.sum((targets_arr - preds_arr) ** 2)
    ss_tot = np.sum((targets_arr - np.mean(targets_arr)) ** 2)
    r2_raw = 1.0 - (ss_res / (ss_tot + 1e-8))

    # 2. Thermodynamic Gibbs-Helmholtz Physical Unit Conversion (kcal/mol -> °C)
    # Delta T_m = (1 / Delta S_m) * (-DDG) + T_m_offset  [1/Delta S_m ≈ 2.1 °C/(kcal/mol), offset ≈ 2.5 °C]
    phys_preds = 2.1 * preds_arr + 2.5
    ss_res_phys = np.sum((targets_arr - phys_preds) ** 2)
    r2_physical = 1.0 - (ss_res_phys / (ss_tot + 1e-8))

    # 3. Zero-Leakage Leave-One-Out (LOO) Calibrated R^2
    r2_loo, loo_preds = leave_one_out_calibrated_r2(preds_arr, targets_arr)

    # 4. Reference Full-fit Calibrated R^2
    slope, intercept = np.polyfit(preds_arr, targets_arr, 1)
    calibrated_preds_leaky = slope * preds_arr + intercept
    ss_res_leaky = np.sum((targets_arr - calibrated_preds_leaky) ** 2)
    r2_calibrated_leaky = 1.0 - (ss_res_leaky / (ss_tot + 1e-8))

    spearman_rho, p_val = spearmanr(preds_arr, targets_arr)
    pearson_r, _ = pearsonr(preds_arr, targets_arr)

    print("-" * 65)
    print("BENCHMARK METRICS RESULTS:")
    print(f"  - Raw R^2 (no calibration)                : {r2_raw:.4f}")
    print(f"  - Thermodynamic Physical R^2 (kcal/mol->°C): {r2_physical:.4f}")
    print(f"  - Leave-One-Out Calibrated R^2 (HONEST)    : {r2_loo:.4f}")
    print(f"  - [reference only, leaky] Full-fit Cal. R^2: {r2_calibrated_leaky:.4f}  <- do not report this")
    print(f"  - p-value                                  : {p_val:.4e}")
    print(f"  - Spearman Correlation (rho)                : {spearman_rho:.4f}")
    print(f"  - Pearson Correlation (r)                  : {pearson_r:.4f}")
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
        "p_value": p_val, "spearman_rho": spearman_rho, "pearson_r": pearson_r,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_s2648.pt",
                     help="Path to a model checkpoint.")
    args = ap.parse_args()
    run_benchmark(args.checkpoint)