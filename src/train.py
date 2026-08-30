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
    """
    Returns primary checkpoint directory ('checkpoints') and optional Google Drive
    backup directory if mounted on Google Colab ('/content/drive/MyDrive/PETase_Checkpoints').
    """
    primary_dir = "checkpoints"
    os.makedirs(primary_dir, exist_ok=True)
    
    drive_dir = None
    if os.path.exists("/content/drive/MyDrive"):
        drive_dir = "/content/drive/MyDrive/PETase_Checkpoints"
        os.makedirs(drive_dir, exist_ok=True)
        
    return primary_dir, drive_dir


def save_checkpoint_dual(state_dict, filename):
    """Saves checkpoint weights to local checkpoints/ and Google Drive (if mounted)."""
    primary_dir, drive_dir = get_checkpoint_dirs()
    
    primary_path = os.path.join(primary_dir, filename)
    torch.save(state_dict, primary_path)
    
    if drive_dir is not None:
        drive_path = os.path.join(drive_dir, filename)
        try:
            torch.save(state_dict, drive_path)
        except Exception as e:
            print(f"[checkpoint] Warning: Failed to sync to Google Drive: {e}")


def custom_collate(batch):
    graph_datas = [item[0] for item in batch]
    target_scores = torch.stack([item[1] for item in batch], dim=0).view(-1)
    mutation_poses = [item[2] for item in batch]
    shield_masks = [item[3] for item in batch]
    protein_keys = [item[4] for item in batch]
    source_tags = [item[5] for item in batch]
    return graph_datas, target_scores, mutation_poses, shield_masks, protein_keys, source_tags


def custom_composite_loss(preds, targets, node_preds_list, shield_masks, alpha=0.02, beta=0.05, margin=0.2):
    mse_loss = nn.MSELoss()(preds, targets)
    n = preds.size(0)
    if n > 1:
        preds_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
        targets_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        target_sign = torch.sign(targets_diff)
        ranking_loss = torch.relu(-target_sign * preds_diff + margin)
        mask = ~torch.eye(n, dtype=torch.bool, device=preds.device)
        pairwise_loss = ranking_loss[mask].mean()
    else:
        pairwise_loss = torch.tensor(0.0, device=preds.device)

    shield_penalties = []
    for node_preds, mask in zip(node_preds_list, shield_masks):
        if mask is not None and mask.sum() > 0:
            shielded_preds = node_preds[mask]
            shield_penalties.append(torch.mean(torch.relu(-shielded_preds)))
    shield_penalty = torch.stack(shield_penalties).mean() if shield_penalties else torch.tensor(0.0, device=preds.device)

    return mse_loss + (alpha * pairwise_loss) + (beta * shield_penalty)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_targets = 0.0, [], []
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
            loss = custom_composite_loss(preds, target_scores, node_preds_list, shield_masks)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(target_scores.detach().cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    spearman_rho, _ = spearmanr(all_preds, all_targets) if len(set(all_targets)) > 1 else (float('nan'), None)
    pearson_r, _ = pearsonr(all_preds, all_targets) if len(set(all_targets)) > 1 else (float('nan'), None)
    return avg_loss, spearman_rho, pearson_r


def pretrain(epochs=200, batch_size=16, lr=5e-4, augment_combinations=True, resume=False):
    """
    PHASE 1: pretrain on the general S2648 corpus (+inverse mutations, +
    synthetic multi-point combos as weak/approximate augmentation).
    Auto-syncs checkpoints to Google Drive every 20 epochs and on best loss.
    """
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
        registry=registry,
    )
    print(f"\nPretraining set: {len(dataset)} rows across {len(dataset.base_graphs)} protein/chain graphs")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)
    
    # Resume checkpoint support if requested
    resume_path = os.path.join(primary_dir, "pretrained_s2648_latest.pt")
    if resume and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path))
        print(f"[resume] Loaded existing weights from {resume_path}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss, rho, r = run_epoch(model, loader, optimizer)
        scheduler.step()
        
        saved = ""
        # Save best model checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint_dual(model.state_dict(), "pretrained_s2648.pt")
            saved = " -> [SAVED BEST]"
            
        # Save latest checkpoint every epoch for safe resume
        save_checkpoint_dual(model.state_dict(), "pretrained_s2648_latest.pt")
        
        # Periodic Google Drive backup every 20 epochs
        if epoch % 20 == 0:
            save_checkpoint_dual(model.state_dict(), f"pretrained_s2648_epoch_{epoch:03d}.pt")
            saved += f" [SYNCED EPOCH {epoch}]"

        if epoch % 5 == 0 or epoch == 1 or "SAVED" in saved:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Spearman rho: {rho:.4f} | Pearson r: {r:.4f}{saved}")

    print(f"\nPretraining complete. Best checkpoint saved to checkpoints/pretrained_s2648.pt")
    return "checkpoints/pretrained_s2648.pt"


def zero_shot_eval_note():
    print("\nTo evaluate the pretrained model zero-shot on the PETase benchmark, run:")
    print("  python -m src.benchmark_eval --checkpoint checkpoints/pretrained_s2648.pt")


def calibrate(pretrained_checkpoint="checkpoints/pretrained_s2648.pt", epochs=30, lr=1e-4):
    """
    PHASE 2: Light calibration (regression head only) on real PETase data.
    Freezes EGNN backbone parameters to prevent catastrophic forgetting.
    """
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
        # Fall back to drive checkpoint if local is missing
        primary_dir, drive_dir = get_checkpoint_dirs()
        if drive_dir and os.path.exists(os.path.join(drive_dir, "pretrained_s2648.pt")):
            pretrained_checkpoint = os.path.join(drive_dir, "pretrained_s2648.pt")

    model.load_state_dict(torch.load(pretrained_checkpoint))

    # Freeze everything except the regression head
    for name, param in model.named_parameters():
        param.requires_grad = "regression_head" in name
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable)} "
          f"(out of {sum(p.numel() for p in model.parameters())} total)")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    optimizer = AdamW(trainable, lr=lr, weight_decay=1e-2)

    for epoch in range(1, epochs + 1):
        loss, rho, r = run_epoch(model, loader, optimizer)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss:.4f} | Spearman rho: {rho:.4f} | Pearson r: {r:.4f}")

    save_checkpoint_dual(model.state_dict(), "calibrated_petase.pt")
    print(f"\nCalibration complete. Saved: checkpoints/calibrated_petase.pt")
    print("Evaluate with: python -m src.benchmark_eval --checkpoint checkpoints/calibrated_petase.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pretrain", "calibrate"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Resume pretraining from latest checkpoint if available")
    args = ap.parse_args()

    if args.phase == "pretrain":
        pretrain(epochs=args.epochs or 200, resume=args.resume)
        zero_shot_eval_note()
    else:
        calibrate(epochs=args.epochs or 30)