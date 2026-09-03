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
    """
    Thin wrapper around PETaseMutationDataset that exposes only the rows
    whose protein_key is in `allowed_protein_keys` (and, optionally, whose
    source_tag is in `allowed_source_tags`). Does NOT touch dataset.py --
    reuses the same base_graphs and __getitem__ logic, just filters which
    indices are visible.

    Used to carve out a held-out-protein validation split: proteins in the
    validation set are fully absent from the training subset, and the
    validation subset is restricted to real rows only (no inverse/synthetic
    augmentation), so validation Spearman reflects genuine generalization
    to unseen protein backbones -- not training-set model selection.
    """

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


def custom_composite_loss(preds, targets, node_preds_list, shield_masks, protein_keys, alpha=0.02, beta=0.05, margin=0.2):
    # 1. Scale-anchoring MSE Loss
    mse_loss = nn.MSELoss()(preds, targets)

    # 2. Pairwise Margin Ranking Loss (RESTRICTED TO SAME-PROTEIN PAIRS ONLY)
    n = preds.size(0)
    if n > 1:
        preds_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
        targets_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        target_sign = torch.sign(targets_diff)

        ranking_loss = torch.relu(-target_sign * preds_diff + margin)

        # TIED-TARGET FIX: when two rows have equal real stability scores,
        # target_sign is 0, which previously still produced a constant
        # `margin` penalty even though there's no real ranking signal to
        # enforce between them. Zero those pairs out explicitly.
        ranking_loss = ranking_loss * (target_sign != 0).float()

        # SAME-PROTEIN MASK: Only rank mutations belonging to the exact same protein backbone!
        pkey_arr = np.array(protein_keys)
        same_protein = torch.tensor(pkey_arr[:, None] == pkey_arr[None, :], dtype=torch.bool, device=preds.device)
        non_self_mask = ~torch.eye(n, dtype=torch.bool, device=preds.device)

        valid_pair_mask = same_protein & non_self_mask

        if valid_pair_mask.any():
            pairwise_loss = ranking_loss[valid_pair_mask].mean()
        else:
            pairwise_loss = torch.tensor(0.0, device=preds.device)
    else:
        pairwise_loss = torch.tensor(0.0, device=preds.device)
        valid_pair_mask = None

    # 3. Active Site Shield Penalty (Exclusively active on hydrolases with non-zero shield masks)
    shield_penalties = []
    for node_preds, mask in zip(node_preds_list, shield_masks):
        if mask is not None and mask.sum() > 0:
            shielded_preds = node_preds[mask]
            shield_penalties.append(torch.mean(torch.relu(-shielded_preds)))

    shield_penalty = torch.stack(shield_penalties).mean() if shield_penalties else torch.tensor(0.0, device=preds.device)

    total_loss = mse_loss + (alpha * pairwise_loss) + (beta * shield_penalty)
    n_valid_pairs = int(valid_pair_mask.sum().item()) if valid_pair_mask is not None else 0
    return total_loss, n_valid_pairs


def run_epoch(model, loader, optimizer=None, track_pairs=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_targets = 0.0, [], []
    real_preds, real_targets = [], []
    pair_counts = [] if track_pairs else None

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for graph_datas, target_scores, mutation_poses, shield_masks, protein_keys, source_tags in loader:
            if is_train:
                optimizer.zero_grad()
            preds_list, node_preds_list = [], []
            for graph_data, pos in zip(graph_datas, mutation_poses):
                p, np_pred = model(graph_data, pos)
                preds_list.append(p)
                node_preds_list.append(np_pred)
            preds = torch.cat(preds_list, dim=0)

            loss, n_valid_pairs = custom_composite_loss(
                preds, target_scores, node_preds_list, shield_masks, protein_keys
            )
            if track_pairs:
                pair_counts.append(n_valid_pairs)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            p_np = preds.detach().cpu().numpy()
            t_np = target_scores.detach().cpu().numpy()

            all_preds.extend(p_np)
            all_targets.extend(t_np)

            for i, stag in enumerate(source_tags):
                if stag == "real":
                    real_preds.append(p_np[i])
                    real_targets.append(t_np[i])

    avg_loss = total_loss / max(len(loader), 1)
    spearman_rho, _ = spearmanr(all_preds, all_targets) if len(set(all_targets)) > 1 else (float('nan'), None)
    pearson_r, _ = pearsonr(all_preds, all_targets) if len(set(all_targets)) > 1 else (float('nan'), None)
    real_rho, _ = spearmanr(real_preds, real_targets) if len(set(real_targets)) > 1 else (float('nan'), None)

    avg_pairs = (sum(pair_counts) / len(pair_counts)) if pair_counts else None

    return avg_loss, spearman_rho, pearson_r, real_rho, avg_pairs


def pretrain(epochs=200, batch_size=16, lr=5e-4, augment_combinations=True, resume=False,
             holdout_frac=0.12):
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
        max_synthetic_ratio=1.0,
        registry=registry,
    )
    print(f"\nPretraining set: {len(dataset)} total rows across {len(dataset.base_graphs)} protein/chain graphs")

    # HELD-OUT PROTEIN SPLIT: carve out whole protein backbones (not
    # individual rows) so the model never sees any row -- real, inverse, or
    # synthetic -- from a held-out protein during training. This is the
    # actual generalization signal; row-level splits would leak information
    # across mutations of the same protein.
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
    if len(val_subset) < 5:
        print("[split] WARNING: fewer than 5 real held-out validation rows -- "
              "validation Spearman will be noisy. Consider a larger holdout_frac "
              "or check that held-out proteins actually have real (non-synthetic) rows.")

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
    for epoch in range(1, epochs + 1):
        train_loss, rho_overall, r_overall, rho_real_train, avg_pairs = run_epoch(
            model, train_loader, optimizer, track_pairs=True
        )
        val_loss, val_rho_overall, val_r_overall, val_rho_real, _ = run_epoch(
            model, val_loader, optimizer=None, track_pairs=False
        )
        scheduler.step()

        # val_rho_overall == val_rho_real here since val_loader is real-only,
        # but both are returned for clarity/consistency with the train side.
        saved = ""
        if not np.isnan(val_rho_overall) and val_rho_overall > best_val_rho:
            best_val_rho = val_rho_overall
            save_checkpoint_dual(model.state_dict(), "pretrained_s2648.pt")
            saved = " -> [SAVED BEST HELD-OUT VAL RHO]"

        save_checkpoint_dual(model.state_dict(), "pretrained_s2648_latest.pt")

        if epoch % 20 == 0:
            save_checkpoint_dual(model.state_dict(), f"pretrained_s2648_epoch_{epoch:03d}.pt")
            saved += f" [SYNCED EPOCH {epoch}]"

        if epoch % 5 == 0 or epoch == 1 or "SAVED" in saved:
            current_lr = scheduler.get_last_lr()[0]
            pair_str = f"{avg_pairs:.2f}" if avg_pairs is not None else "n/a"
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Train ρ: {rho_overall:.4f} | Train-real ρ: {rho_real_train:.4f} | "
                  f"HELD-OUT VAL ρ: {val_rho_overall:.4f} | "
                  f"Avg same-protein pairs/batch: {pair_str} | LR: {current_lr:.6f}{saved}")

    print(f"\nPretraining complete. Best held-out-val checkpoint saved to checkpoints/pretrained_s2648.pt "
          f"(best val rho: {best_val_rho:.4f})")
    return "checkpoints/pretrained_s2648.pt"


def zero_shot_eval_note():
    print("\nTo evaluate the pretrained model zero-shot on the PETase benchmark, run:")
    print("  python -m src.benchmark_eval --checkpoint checkpoints/pretrained_s2648.pt")


def calibrate(pretrained_checkpoint="checkpoints/pretrained_s2648.pt", epochs=30, lr=1e-4):
    print("=" * 70)
    print("PHASE 2: LIGHT CALIBRATION (regression head only) on real PETase data")
    print("=" * 70)

    registry = build_registry(verbose=False)
    dataset = PETaseMutationDataset(
        csv_paths=["data/mutations_verified_stability.csv"],
        augment_inverse=True, augment_combinations=False,
        registry=registry,
    )
    print(f"Calibration set: {len(dataset)} rows (real experimental data only)")

    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)

    if not os.path.exists(pretrained_checkpoint):
        primary_dir, drive_dir = get_checkpoint_dirs()
        if drive_dir and os.path.exists(os.path.join(drive_dir, "pretrained_s2648.pt")):
            pretrained_checkpoint = os.path.join(drive_dir, "pretrained_s2648.pt")

    model.load_state_dict(torch.load(pretrained_checkpoint))

    for name, param in model.named_parameters():
        param.requires_grad = "regression_head" in name
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable)} "
          f"(out of {sum(p.numel() for p in model.parameters())} total)")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    optimizer = AdamW(trainable, lr=lr, weight_decay=1e-2)

    for epoch in range(1, epochs + 1):
        loss, rho_overall, r_overall, rho_real, _ = run_epoch(model, loader, optimizer, track_pairs=False)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss:.4f} | Spearman rho: {rho_overall:.4f} | Pearson r: {r_overall:.4f}")

    save_checkpoint_dual(model.state_dict(), "calibrated_petase.pt")
    print(f"\nCalibration complete. Saved: checkpoints/calibrated_petase.pt")
    print("Evaluate with: python -m src.benchmark_eval --checkpoint checkpoints/calibrated_petase.pt")
    print("NOTE: with only ~4 real calibration rows, this loop has no held-out split of its own -- "
          "report leave-one-out Spearman separately (not yet implemented here) rather than treating "
          "this training-set rho as a generalization claim.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pretrain", "calibrate"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--holdout_frac", type=float, default=0.12)
    ap.add_argument("--resume", action="store_true", help="Resume pretraining from latest checkpoint")
    args = ap.parse_args()

    if args.phase == "pretrain":
        pretrain(epochs=args.epochs or 200, lr=args.lr or 5e-4, resume=args.resume,
                  holdout_frac=args.holdout_frac)
        zero_shot_eval_note()
    else:
        calibrate(epochs=args.epochs or 30, lr=args.lr or 1e-4)