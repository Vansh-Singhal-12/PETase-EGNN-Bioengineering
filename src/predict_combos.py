import os
import argparse
import csv
import torch

from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN
from src.protein_registry import build_registry


def load_single_score_lookup(round1_ranked_csv):
    """Returns dict: (position, mutation_type) -> predicted_score, from Round 1 results."""
    lookup = {}
    with open(round1_ranked_csv) as f:
        for r in csv.DictReader(f):
            pos = int(r["position_idx"].split(",")[0])
            lookup[(pos, r["mutation_type"])] = float(r["predicted_score"])
    return lookup


def score_combos(checkpoint_path, combo_csv_path, round1_ranked_csv, output_path):
    registry = build_registry(verbose=False)
    dataset = PETaseMutationDataset(
        csv_paths=[combo_csv_path],
        augment_inverse=False, augment_combinations=False,
        registry=registry,
    )
    print(f"[predict_combos] Loaded {len(dataset)} pairwise combos from '{combo_csv_path}'")

    single_lookup = load_single_score_lookup(round1_ranked_csv)

    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    results = []
    with torch.no_grad():
        for i in range(len(dataset)):
            graph_data, _, mutation_pos, shield_mask, protein_key, source_tag = dataset[i]
            pos_list, _, wt_list, mut_list, _, _ = dataset.items[i]

            combo_pred, _ = model(graph_data, mutation_pos, use_interaction=True)
            combo_score = combo_pred.item()

            real_positions = [p + registry[protein_key]["first_resolved_residue"] for p in pos_list]

            single_scores = []
            missing = False
            for p, mut in zip(real_positions, mut_list):
                s = single_lookup.get((p, mut))
                if s is None:
                    missing = True
                    break
                single_scores.append(s)

            if missing:
                additive_expectation = None
                epistasis_score = None
            else:
                additive_expectation = sum(single_scores)
                epistasis_score = combo_score - additive_expectation

            results.append({
                "wild_type": ";".join(wt_list),
                "mutation_type": ";".join(mut_list),
                "position_idx": ",".join(str(p) for p in real_positions),
                "combo_score": combo_score,
                "additive_expectation": additive_expectation,
                "epistasis_score": epistasis_score,
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "combo_score", "additive_expectation", "epistasis_score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[predict_combos] Scored {len(results)} combos -> {output_path}\n")

    by_score = sorted(results, key=lambda r: r["combo_score"], reverse=True)
    print("[predict_combos] TOP 15 by raw combo score:")
    for r in by_score[:15]:
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"combo: {r['combo_score']:.3f} | epistasis: {r['epistasis_score']}")

    epistasis_valid = [r for r in results if r["epistasis_score"] is not None]
    by_epistasis = sorted(epistasis_valid, key=lambda r: r["epistasis_score"], reverse=True)
    print("\n[predict_combos] TOP 15 by EPISTASIS score (combo >> sum of individual parts):")
    for r in by_epistasis[:15]:
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"combo: {r['combo_score']:.3f} | additive expectation: {r['additive_expectation']:.3f} | "
              f"epistasis: {r['epistasis_score']:.3f}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_s2648.pt")
    ap.add_argument("--combo_csv", type=str, default="data/screening_round2_pairwise.csv")
    ap.add_argument("--round1_ranked", type=str, default="results/round1_ranked.csv")
    ap.add_argument("--output", type=str, default="results/round2_combo_ranked.csv")
    args = ap.parse_args()

    score_combos(args.checkpoint, args.combo_csv, args.round1_ranked, args.output)