import os
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

# Deterministic Seeding & Backend Lock
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_collate(batch):
    graph_datas = [item[0] for item in batch]
    target_scores = torch.stack([item[1] for item in batch], dim=0).view(-1)
    mutation_poses = [item[2] for item in batch]
    shield_masks = [item[3] for item in batch]
    return graph_datas, target_scores, mutation_poses, shield_masks


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
            
    if len(shield_penalties) > 0:
        shield_penalty = torch.stack(shield_penalties).mean()
    else:
        shield_penalty = torch.tensor(0.0, device=preds.device)
        
    total_loss = mse_loss + (alpha * pairwise_loss) + (beta * shield_penalty)
    return total_loss


def run_training():
    print("Setting up EGNN Pipeline Smoke Test on Verified Literature Rows...")
    os.makedirs("checkpoints", exist_ok=True)
    
    dataset = PETaseMutationDataset(pdb_path="data/6eqe.pdb", csv_path="data/mutations_clean.csv", augment_inverse=True)
    print(f"[Data Augmentation] Base rows: {len(dataset.df)} -> Augmented rows: {len(dataset)}")
    
    # Dynamic batch size guard for small datasets
    batch_size = min(16, max(2, len(dataset) // 2))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    print(f"Batch Size: {batch_size} | Total Batches per Epoch: {len(loader)}")
    
    model = PETaseStabilityEGNN(in_dim=8, emb_dim=32, dropout=0.1)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    best_spearman = -1.0
    
    print("-" * 75)
    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []
        
        for batch in loader:
            optimizer.zero_grad()
            
            graph_datas, target_scores, mutation_poses, shield_masks = batch
            
            preds_list = []
            node_preds_list = []
            
            for graph_data, pos in zip(graph_datas, mutation_poses):
                p, np_pred = model(graph_data, pos)
                preds_list.append(p)
                node_preds_list.append(np_pred)
                
            preds = torch.cat(preds_list, dim=0)
            
            loss = custom_composite_loss(preds, target_scores, node_preds_list, shield_masks, alpha=0.02, beta=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(target_scores.detach().cpu().numpy())
            
        scheduler.step()
        
        avg_loss = total_loss / len(loader)
        spearman_rho, _ = spearmanr(all_preds, all_targets)
        pearson_r, _ = pearsonr(all_preds, all_targets)
        current_lr = scheduler.get_last_lr()[0]
        
        saved_str = ""
        if spearman_rho > best_spearman:
            best_spearman = spearman_rho
            torch.save(model.state_dict(), "checkpoints/best_egnn.pt")
            saved_str = " ➔ [MODEL SAVED]"
            
        if epoch % 10 == 0 or epoch == 1 or saved_str != "":
            print(f"Epoch {epoch:03d}/100 | Loss: {avg_loss:.4f} | Spearman ρ: {spearman_rho:.4f} | Pearson r: {pearson_r:.4f} | LR: {current_lr:.6f}{saved_str}")

    print("-" * 75)
    print(f"Smoke test training complete. Best Spearman ρ achieved: {best_spearman:.4f}")

if __name__ == "__main__":
    run_training()