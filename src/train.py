import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr

from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN
from src.protein_registry import build_registry

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def get_checkpoint_dirs():
    primary_dir = "checkpoints"
    os.makedirs(primary_dir, exist_ok=True)
    drive_dir = None
    if os.path.exists("/content/drive/MyDrive"):
        drive_dir = "/content/drive/MyDrive/PETase_Checkpoints"
        os.makedirs(drive_dir, exist_ok=True)
    return primary_dir, drive_dir


def save_checkpoint_dual(state_dict, filename):
    primary_dir, drive_dir = get_checkpoint_dirs()
    primary_path = os.path.join(primary_dir, filename)
    torch.save(state_dict, primary_path)
    if drive_dir is not None:
        drive_path = os.path.join(drive_dir, filename)
        try:
            torch.save(state_dict, drive_path)
        except Exception as e:
            print(f"[checkpoint] Warning: Failed to sync to Google Drive: {e}")


class ProteinFilteredSubset:
    def __init__(self, dataset, allowed_protein_keys, allowed_source_tags=None):
        self.dataset = dataset
        self.indices = [
            i for i, item in enumerate(dataset.items)
            if item[4] in allowed_protein_keys
            and (allowed_source_tags is None or item[5] in allowed_source_tags)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def custom_collate(batch):
    graph_datas = [item[0] for item in batch]
    target_scores = torch.stack([item[1] for item in batch], dim=0).view(-1)
    mutation_poses = [item[2] for item in batch]
    shield_masks = [item[3] for item in batch]
    protein_keys = [item[4] for item in batch]
    source_tags = [item[5] for item in batch]
    return graph_datas, target_scores, mutation_poses, shield_masks, protein_keys, source_tags


def custom_composite_loss(mu, log_var, targets, node_preds_list, shield_masks, protein_keys, source_tags,
                          alpha=0.02, beta=0.05, margin=0.2, synthetic_weight=0.5):
    sample_weights = torch.tensor(
        [synthetic_weight if "synthetic" in tag else 1.0 for tag in source_tags],
        dtype=torch.float, device=mu.device
    )

    # 1. Gaussian Negative Log-Likelihood (replaces plain weighted MSE).
    # var = exp(log_var); loss = 0.5*log(var) + 0.5*(target-mu)^2/var.
    # This rewards the model for being HONESTLY uncertain: a wrong
    # prediction with high predicted variance is penalized less than a
    # wrong prediction made with false confidence (low variance), and a
    # correct prediction with unnecessarily high variance is also
    # penalized -- the model is trained to match its confidence to its
    # actual accuracy, not just to minimize raw error.
    var = torch.exp(log_var) + 1e-6
    nll_per_sample = 0.5 * torch.log(var) + 0.5 * (targets - mu) ** 2 / var
    nll_loss = (sample_weights * nll_per_sample).mean()

    # 2. Pairwise Margin Ranking Loss -- now operates on mu (the mean
    # prediction), exactly as it operated on the single output before.
    # Ranking doesn't use log_var at all; it's purely about getting the
    # relative order of predicted means right.
    n = mu.size(0)
    if n > 1:
        preds_diff = mu.unsqueeze(1) - mu.unsqueeze(0)
        targets_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        target_sign = torch.sign(targets_diff)
        ranking_loss = torch.relu(-target_sign * preds_diff + margin)
        ranking_loss = ranking_loss * (target_sign != 0).float()

        pair_weights = torch.min(sample_weights.unsqueeze(1), sample_weights.unsqueeze(0))
        weighted_ranking_loss = ranking_loss * pair_weights

        pkey_arr = np.array(protein_keys)
        same_protein = torch.tensor(pkey_arr[:, None] == pkey_arr[None, :], dtype=torch.bool, device=mu.device)
        non_self_mask = ~torch.eye(n, dtype=torch.bool, device=mu.device)
        valid_pair_mask = same_protein & non_self_mask

        if valid_pair_mask.any():
            pairwise_loss = weighted_ranking_loss[valid_pair_mask].mean()
        else:
            pairwise_loss = torch.tensor(0.0, device=mu.device)
    else:
        pairwise_loss = torch.tensor(0.0, device=mu.device)
        valid_pair_mask = None

    # 3. Active Site Shield Penalty -- unchanged, still operates on
    # node_preds (per-node readout head, untouched by the mu/log_var change).
    shield_penalties = []
    for node_preds, mask in zip(node_preds_list, shield_masks):
        if mask is not None and mask.sum() > 0:
            shielded_preds = node_preds[mask]
            shield_penalties.append(torch.mean(torch.relu(-shielded_preds)))
    shield_penalty = torch.stack(shield_penalties).mean() if shield_penalties else torch.tensor(0.0, device=mu.device)

    total_loss = nll_loss + (alpha * pairwise_loss) + (beta * shield_penalty)
    n_valid_pairs = int(valid_pair_mask.sum().item()) if valid_pair_mask is not None else 0
    return total_loss, n_valid_pairs


def run_epoch(model, loader, optimizer=None, track_pairs=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_mu, all_targets, all_sigma = 0.0, [], [], []
    real_mu, real_targets = [], []
    pair_counts = [] if track_pairs else None

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for graph_datas, target_scores, mutation_poses, shield_masks, protein_keys, source_tags in loader:
            if is_train:
                optimizer.zero_grad()

            mu_list, log_var_list, node_preds_list = [], [], []
            for graph_data, pos, stag in zip(graph_datas, mutation_poses, source_tags):
                use_interact = stag in ("real", "real_inverse")

                if is_train:
                    g_jitter = graph_data.clone()
                    g_jitter.pos = g_jitter.pos + torch.randn_like(g_jitter.pos) * 0.03
                    mu_i, log_var_i, np_pred = model(g_jitter, pos, use_interaction=use_interact)
                else:
                    mu_i, log_var_i, np_pred = model(graph_data, pos, use_interaction=use_interact)

                mu_list.append(mu_i)
                log_var_list.append(log_var_i)
                node_preds_list.append(np_pred)

            mu = torch.cat(mu_list, dim=0)
            log_var = torch.cat(log_var_list, dim=0)

            loss, n_valid_pairs = custom_composite_loss(
                mu, log_var, target_scores, node_preds_list, shield_masks, protein_keys, source_tags
            )
            if track_pairs:
                pair_counts.append(n_valid_pairs)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            mu_np = mu.detach().cpu().numpy()
            sigma_np = torch.exp(0.5 * log_var).detach().cpu().numpy()
            t_np = target_scores.detach().cpu().numpy()

            all_mu.extend(mu_np)
            all_sigma.extend(sigma_np)
            all_targets.extend(t_np)

            for i, stag in enumerate(source_tags):
                if stag == "real":
                    real_mu.append(mu_np[i])
                    real_targets.append(t_np[i])

    avg_loss = total_loss / max(len(loader), 1)
    spearman_rho, _ = spearmanr(all_mu, all_targets) if len(set(all_targets)) > 1 else (float('nan'), None)
    pearson_r, _ = pearsonr(all_mu, all_targets) if len(set(all_targets)) > 1 else (float('nan'), None)
    real_rho, _ = spearmanr(real_mu, real_targets) if len(set(real_targets)) > 1 else (float('nan'), None)
    mean_sigma = float(np.mean(all_sigma)) if all_sigma else float('nan')

    avg_pairs = (sum(pair_counts) / len(pair_counts)) if pair_counts else None

    return avg_loss, spearman_rho, pearson_r, real_rho, avg_pairs, mean_sigma


def pretrain(epochs=200, batch_size=16, lr=5e-4, augment_combinations=True, resume=False,
             holdout_frac=0.12, patience=20):
    print("=" * 70)
    print("PHASE 1: PRETRAINING on S2648 general protein-stability corpus")
    print("=" * 70)

    primary_dir, drive_dir = get_checkpoint_dirs()
    if drive_dir:
        print(f"[Google Drive Sync] Active -> Backup path: {drive_dir}")

    registry = build_registry(verbose=True)
    dataset = PETaseMutationDataset(
        csv_paths=["data/mutations_s2648_pretraining.csv"],
        augment_inverse=True, augment_combinations=augment_combinations,
        max_synthetic_ratio=0.25,
        registry=registry,
    )
    print(f"\nPretraining set: {len(dataset)} total rows across {len(dataset.base_graphs)} protein/chain graphs")

    all_protein_keys = sorted(dataset.df["protein_key"].unique().tolist())
    shuffled_keys = all_protein_keys[:]
    random.Random(RANDOM_SEED).shuffle(shuffled_keys)
    n_holdout = max(1, int(round(holdout_frac * len(shuffled_keys))))
    holdout_proteins = set(shuffled_keys[:n_holdout])
    train_proteins = set(shuffled_keys[n_holdout:])

    print(f"[split] {len(train_proteins)} proteins for training, "
          f"{len(holdout_proteins)} held out entirely for validation "
          f"({holdout_frac*100:.0f}% target)")

    train_subset = ProteinFilteredSubset(dataset, train_proteins)
    val_subset = ProteinFilteredSubset(dataset, holdout_proteins, allowed_source_tags={"real"})
    print(f"[split] train rows: {len(train_subset)}, held-out validation rows (real-only): {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)

    resume_path = os.path.join(primary_dir, "pretrained_s2648_latest.pt")
    if resume and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path))
        print(f"[resume] Loaded existing weights from {resume_path}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_rho = -1.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, rho_overall, r_overall, rho_real_train, avg_pairs, train_sigma = run_epoch(
            model, train_loader, optimizer, track_pairs=True
        )
        val_loss, val_rho_overall, val_r_overall, val_rho_real, _, val_sigma = run_epoch(
            model, val_loader, optimizer=None, track_pairs=False
        )
        scheduler.step()

        saved = ""
        if not np.isnan(val_rho_overall) and val_rho_overall > best_val_rho:
            best_val_rho = val_rho_overall
            patience_counter = 0
            save_checkpoint_dual(model.state_dict(), "pretrained_s2648.pt")
            saved = " -> [SAVED BEST HELD-OUT VAL RHO]"
        else:
            patience_counter += 1

        save_checkpoint_dual(model.state_dict(), "pretrained_s2648_latest.pt")

        if epoch % 20 == 0:
            save_checkpoint_dual(model.state_dict(), f"pretrained_s2648_epoch_{epoch:03d}.pt")
            saved += f" [SYNCED EPOCH {epoch}]"

        current_lr = scheduler.get_last_lr()[0]
        pair_str = f"{avg_pairs:.2f}" if avg_pairs is not None else "n/a"
        print(f"Epoch {epoch:03d}/{epochs} | Loss: {train_loss:.4f} | "
              f"Train ρ: {rho_overall:.4f} | Train-real ρ: {rho_real_train:.4f} | "
              f"HELD-OUT VAL ρ: {val_rho_overall:.4f} | "
              f"Train σ̄: {train_sigma:.4f} | Val σ̄: {val_sigma:.4f} | "
              f"Avg same-protein pairs/batch: {pair_str} | LR: {current_lr:.6f}{saved}")

        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement in held-out validation rho for {patience} consecutive epochs. Stopping pretraining at epoch {epoch:03d}.")
            break

    print(f"\nPretraining complete. Best held-out-val checkpoint saved to checkpoints/pretrained_s2648.pt "
          f"(best val rho: {best_val_rho:.4f})")
    return "checkpoints/pretrained_s2648.pt"


def zero_shot_eval_note():
    print("\nTo evaluate the pretrained model zero-shot on the PETase benchmark, run:")
    print("  python -m src.benchmark_eval --checkpoint checkpoints/pretrained_s2648.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pretrain"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--holdout_frac", type=float, default=0.12)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    pretrain(epochs=args.epochs or 200, lr=args.lr or 5e-4, resume=args.resume,
              holdout_frac=args.holdout_frac, patience=args.patience)
    zero_shot_eval_note()