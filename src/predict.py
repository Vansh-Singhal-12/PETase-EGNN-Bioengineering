import os
import argparse
import csv
import torch
import numpy as np
from scipy.stats import mannwhitneyu

from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN
from src.protein_registry import build_registry

RANDOM_SEED = 42


def score_library(checkpoint_path, library_csv_path, output_path, top_n_for_baseline_test=50):
    registry = build_registry(verbose=False)
    dataset = PETaseMutationDataset(
        csv_paths=[library_csv_path],
        augment_inverse=False, augment_combinations=False,
        registry=registry,
    )
    print(f"[predict] Loaded {len(dataset)} candidates from '{library_csv_path}'")

    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    results = []
    with torch.no_grad():
        for i in range(len(dataset)):
            graph_data, _, mutation_pos, shield_mask, protein_key, source_tag = dataset[i]
            pos_list, _, wt_list, mut_list, _, _ = dataset.items[i]

            # Multi-point candidates get use_interaction=True (these are
            # deliberately constructed Round 2 candidates, not synthetic
            # training augmentation, so the interaction module is
            # appropriate here -- unlike the synthetic_combo rows in
            # training which are excluded from it).
            use_interact = True
            pred, _ = model(graph_data, mutation_pos, use_interaction=use_interact)

            results.append({
                "wild_type": ";".join(wt_list),
                "mutation_type": ";".join(mut_list),
                "position_idx": ",".join(str(p + registry[protein_key]["first_resolved_residue"]) for p in pos_list),
                "predicted_score": pred.item(),
                "n_mutations": len(pos_list),
            })

    results.sort(key=lambda r: r["predicted_score"], reverse=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "predicted_score", "n_mutations"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[predict] Scored {len(results)} candidates, ranked, saved to '{output_path}'")
    print(f"[predict] Top 5:")
    for r in results[:5]:
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"predicted: {r['predicted_score']:.4f}")

    # Random-baseline significance test: are the top-N candidates'
    # predicted scores statistically distinguishable from a random sample
    # of the full library? Mann-Whitney U (non-parametric, doesn't assume
    # normal distribution -- appropriate here since there is no reason to
    # assume predicted scores are normally distributed).
    all_scores = np.array([r["predicted_score"] for r in results])
    top_scores = all_scores[:top_n_for_baseline_test]

    rng = np.random.RandomState(RANDOM_SEED)
    random_idx = rng.choice(len(all_scores), size=min(top_n_for_baseline_test, len(all_scores)), replace=False)
    random_scores = all_scores[random_idx]

    stat, p_value = mannwhitneyu(top_scores, random_scores, alternative='greater')

    print(f"\n[predict] RANDOM-BASELINE SIGNIFICANCE TEST (Mann-Whitney U, one-sided):")
    print(f"    Top {top_n_for_baseline_test} mean predicted score   : {top_scores.mean():.4f}")
    print(f"    Random {len(random_scores)} mean predicted score     : {random_scores.mean():.4f}")
    print(f"    U statistic                                 : {stat:.2f}")
    print(f"    p-value (top > random)                      : {p_value:.4e}")
    if p_value < 0.05:
        print(f"    SIGNIFICANT: top-ranked candidates are not indistinguishable from random selection.")
    else:
        print(f"    NOT SIGNIFICANT: screening did not clearly outperform random selection -- investigate.")

    return output_path, {"p_value": p_value, "top_mean": top_scores.mean(), "random_mean": random_scores.mean()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_s2648.pt")
    ap.add_argument("--library", type=str, required=True,
                     help="Path to the unlabeled candidate CSV (from generate_library.py or generate_combos.py)")
    ap.add_argument("--output", type=str, default="results/screening_ranked.csv")
    ap.add_argument("--top_n_baseline", type=int, default=50)
    args = ap.parse_args()

    score_library(args.checkpoint, args.library, args.output, top_n_for_baseline_test=args.top_n_baseline)