"""
Measures how much a mutation perturbs the model's internal representation
of the catalytic-triad-adjacent region, using the EGNN's well-trained
backbone embeddings (h).

Metric: cosine distance between the WT and mutant embeddings at each
shielded (triad-region) residue, taking the MAX across those residues as
the candidate's score -- the single most-perturbed shielded residue.
Backbone embeddings receive gradient from the ENTIRE training set, so
this is a legitimate use of a validated part of the model.

Tested for real discriminative variance before being treated as usable --
same discipline as every other check in this pipeline.
"""
import argparse
import csv
import torch
import numpy as np

from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN
from src.protein_registry import build_registry


def get_backbone_embeddings(model, graph_data):
    """Runs just the embedding + 2 EGNN layers (not the full forward pass),
    returning the per-node h embeddings after message passing."""
    h = model.embedding(graph_data.x.float())
    pos = graph_data.pos.float()
    edge_index = graph_data.edge_index
    h, pos = model.layer1(h, pos, edge_index)
    h, pos = model.layer2(h, pos, edge_index)
    return h


def score_drift(checkpoint_path, candidates_csv_path, output_path, drift_flag_percentile=90):
    registry = build_registry(verbose=False)
    dataset = PETaseMutationDataset(
        csv_paths=[candidates_csv_path],
        augment_inverse=False, augment_combinations=False,
        registry=registry,
    )

    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Get WT baseline embedding ONCE (unmutated graph) for the shielded residues
    wt_graph = dataset.base_graphs["6EQE"]
    with torch.no_grad():
        wt_h = get_backbone_embeddings(model, wt_graph)
    shield_mask = wt_graph.active_site_shield
    wt_shielded_h = wt_h[shield_mask]  # [n_shielded_residues, 32]

    results = []
    with torch.no_grad():
        for i in range(len(dataset)):
            graph_data, _, mutation_pos, _, protein_key, source_tag = dataset[i]
            pos_list, _, wt_list, mut_list, _, _ = dataset.items[i]

            mut_h = get_backbone_embeddings(model, graph_data)
            mut_shielded_h = mut_h[shield_mask]

            # Cosine distance per shielded residue: 1 - cosine_similarity
            cos_sim = torch.nn.functional.cosine_similarity(wt_shielded_h, mut_shielded_h, dim=-1)
            cos_dist = (1 - cos_sim)
            max_drift = cos_dist.max().item()
            mean_drift = cos_dist.mean().item()

            results.append({
                "wild_type": ";".join(wt_list),
                "mutation_type": ";".join(mut_list),
                "position_idx": ",".join(str(p + registry[protein_key]["first_resolved_residue"]) for p in pos_list),
                "max_triad_embedding_drift": max_drift,
                "mean_triad_embedding_drift": mean_drift,
            })

    drift_values = np.array([r["max_triad_embedding_drift"] for r in results])
    print(f"[triad_drift] Distribution across {len(results)} candidates:")
    print(f"    min={drift_values.min():.5f}  25th pct={np.percentile(drift_values,25):.5f}  "
          f"median={np.median(drift_values):.5f}  75th pct={np.percentile(drift_values,75):.5f}  "
          f"max={drift_values.max():.5f}  std={drift_values.std():.5f}")

    flag_threshold = float(np.percentile(drift_values, drift_flag_percentile))
    n_flagged = 0
    for r in results:
        r["drift_flag"] = r["max_triad_embedding_drift"] >= flag_threshold
        if r["drift_flag"]:
            n_flagged += 1

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "max_triad_embedding_drift", "mean_triad_embedding_drift",
                                                 "drift_flag"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[triad_drift] Flag threshold ({drift_flag_percentile}th pct): {flag_threshold:.5f}")
    print(f"[triad_drift] {n_flagged}/{len(results)} flagged as high-drift (advisory, review not auto-exclude)")
    print(f"[triad_drift] Wrote -> {output_path}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_s2648.pt")
    ap.add_argument("--candidates", type=str, required=True)
    ap.add_argument("--output", type=str, default="results/triad_drift_scores.csv")
    ap.add_argument("--drift_flag_percentile", type=int, default=90)
    args = ap.parse_args()

    score_drift(args.checkpoint, args.candidates, args.output,
                drift_flag_percentile=args.drift_flag_percentile)