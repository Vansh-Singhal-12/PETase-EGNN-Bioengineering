"""
diagnose_zero_shot_failure.py

Two checks, run in order:

1. IN-SAMPLE SANITY CHECK: reload the pretrained checkpoint fresh (exactly
   like benchmark_eval.py does) and evaluate it on a sample of the SAME
   S2648 data it was trained on. Training logs reported Spearman rho ~0.85
   by epoch 110 -- if a fresh eval on that same data now shows near-zero or
   negative correlation, that's a smoking gun: something differs between
   how metrics were computed DURING training vs a fresh reload (e.g. a
   train/eval mode mismatch, or a checkpoint save/load bug), not a
   transfer-learning problem at all.

   If in-sample correlation is still strong (close to what training logs
   showed), the checkpoint and loading code are fine -- the zero-shot
   failure is specifically about generalizing to PETase, not a bug.

2. RAW PREDICTION INSPECTION on the benchmark itself: prints every
   prediction vs target side by side. A clean sign-flip (predictions
   consistently backwards) looks very different from scattered/uncorrelated
   values, and tells us which failure mode we're actually looking at.

USAGE (from repo root):
    python diagnose_zero_shot_failure.py --checkpoint checkpoints/pretrained_s2648_epoch_200.pt
"""
import argparse
import random
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr

from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN
from src.protein_registry import build_registry


def in_sample_check(checkpoint_path, sample_size=150):
    print("=" * 70)
    print("CHECK 1: IN-SAMPLE SANITY -- does the checkpoint still fit its own training data?")
    print("=" * 70)

    registry = build_registry(verbose=False)
    # augment_inverse/combinations OFF here -- we want the RAW real single-
    # point rows only, the cleanest possible signal, not synthetic combos
    dataset = PETaseMutationDataset(
        csv_paths=["data/mutations_s2648_pretraining.csv"],
        augment_inverse=False, augment_combinations=False,
        registry=registry,
    )
    print(f"Full training set: {len(dataset)} real rows. Sampling {sample_size} for this check.")

    random.seed(0)
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))

    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for i in indices:
            graph_data, target_score, mutation_pos, shield_mask, protein_key, source_tag = dataset[i]
            pred, _ = model(graph_data, mutation_pos)
            preds.append(pred.item())
            targets.append(target_score.item())

    rho, p = spearmanr(preds, targets)
    r, _ = pearsonr(preds, targets)
    print(f"\nIn-sample (same data it trained on) Spearman rho: {rho:.4f}, Pearson r: {r:.4f}")
    print("Compare this to the ~0.85 rho reported in training logs around epoch 110.")
    if rho > 0.5:
        print(">> Checkpoint/loading is fine. The zero-shot PETase failure is a real")
        print(">> generalization problem, not a bug in saving/loading/evaluating.")
    else:
        print(">> WARNING: even in-sample correlation is weak. This points to a bug in")
        print(">> checkpoint save/load or eval-mode behavior, not (only) generalization.")
    return rho


def benchmark_prediction_inspection(checkpoint_path):
    print("\n" + "=" * 70)
    print("CHECK 2: RAW PREDICTIONS vs TARGETS on the benchmark itself")
    print("=" * 70)

    registry = build_registry(verbose=False)
    dataset = PETaseMutationDataset(
        csv_paths=["data/benchmark_25.csv"],
        augment_inverse=False, augment_combinations=False,
        registry=registry,
    )

    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    print(f"\n{'Row':<5}{'Target':<12}{'Prediction':<14}{'Positions'}")
    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            graph_data, target_score, mutation_pos, shield_mask, protein_key, source_tag = dataset[i]
            pred, _ = model(graph_data, mutation_pos)
            rows.append((target_score.item(), pred.item(), mutation_pos.tolist()))
            print(f"{i:<5}{target_score.item():<12.3f}{pred.item():<14.4f}{mutation_pos.tolist()}")

    targets = [r[0] for r in rows]
    preds = [r[1] for r in rows]
    print(f"\nTarget range: [{min(targets):.2f}, {max(targets):.2f}]  (all should be positive -- these are curated success stories)")
    print(f"Prediction range: [{min(preds):.2f}, {max(preds):.2f}]")
    n_pred_positive = sum(1 for p in preds if p > 0)
    print(f"Predictions that are positive: {n_pred_positive}/{len(preds)}")
    print("\nIf predictions cluster near a narrow range regardless of target, that's a")
    print("scale/collapse problem. If predictions are broadly negative while ALL targets")
    print("are positive, that's consistent with a real distribution-shift effect (see")
    print("write-up notes) rather than a random/scattered failure.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    args = ap.parse_args()
    in_sample_check(args.checkpoint)
    benchmark_prediction_inspection(args.checkpoint)