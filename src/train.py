import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from scipy.stats import spearmanr, pearsonr
from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN

# Absolute Reproducibility Seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CompositeMarginLoss(nn.Module):
    """
    Custom Multi-Objective Composite Loss Function:
    L_total = L_MSE (Scale Anchor) + alpha * L_MarginRank (Pairwise Ordinal Ranking)
    """
    def __init__(self, margin=0.2, alpha=0.1):
        super(CompositeMarginLoss, self).__init__()
        self.mse_fn = nn.MSELoss()
        self.ranking_fn = nn.MarginRankingLoss(margin=margin)
        self.alpha = alpha

    def forward(self, preds, targets):
        # 1. Scale Anchor MSE Loss
        mse_loss = self.mse_fn(preds, targets)

        # 2. Pairwise Margin Ranking Loss across mini-batch pairs
        n = preds.size(0)
        if n > 1:
            p1 = preds.repeat_interleave(n)
            p2 = preds.repeat(n)
            t1 = targets.repeat_interleave(n)
            t2 = targets.repeat(n)

            target_sign = torch.sign(t1 - t2)
            non_zero_mask = target_sign != 0

            if non_zero_mask.sum() > 0:
                rank_loss = self.ranking_fn(p1[non_zero_mask], p2[non_zero_mask], target_sign[non_zero_mask])
            else:
                rank_loss = torch.tensor(0.0, device=preds.device)
        else:
            rank_loss = torch.tensor(0.0, device=preds.device)

        total_loss = mse_loss + (self.alpha * rank_loss)
        return total_loss

def run_training():
    print("Setting up Mini-Batched EGNN Training Loop with Composite Margin Loss...")

    os.makedirs("checkpoints", exist_ok=True)

    # Load 154-sample dataset (auto-augmented to 308 rows in memory)
    dataset = PETaseMutationDataset(
        pdb_path="data/6eqe.pdb",
        csv_path="data/mutations_clean.csv",
        augment_inverse=True
    )

    model = PETaseStabilityEGNN(num_amino_acids=8, emb_dim=32, radius=10.0)

    # AdamW Optimizer + Cosine Annealing Learning Rate Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    NUM_EPOCHS = 200
    BATCH_SIZE = 16
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    criterion = CompositeMarginLoss(margin=0.2, alpha=0.1)

    best_spearman_rho = -1.0
    patience = 40
    patience_counter = 0

    print(f"Starting mini-batch training run over all {len(dataset)} augmented rows...")
    print(f"Batch Size: {BATCH_SIZE} | Total Batches per Epoch: {(len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print("-" * 75)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle indices per epoch for stochastic gradient descent
        indices = torch.randperm(len(dataset)).tolist()
        
        epoch_preds = []
        epoch_targets = []

        # Mini-batch loop
        for i in range(0, len(dataset), BATCH_SIZE):
            batch_indices = indices[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            
            batch_preds = []
            batch_targets = []

            for idx in batch_indices:
                graph_data, target_score, mutation_pos, active_site_shield = dataset[idx]
                pred = model(graph_data, mutation_pos)
                
                batch_preds.append(pred.view(-1))
                batch_targets.append(target_score.view(-1))

            preds_tensor = torch.cat(batch_preds)
            targets_tensor = torch.cat(batch_targets)

            loss = criterion(preds_tensor, targets_tensor)
            loss.backward()

            # Gradient clipping to prevent sudden gradient spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * len(batch_indices)
            
            epoch_preds.extend(preds_tensor.detach().cpu().tolist())
            epoch_targets.extend(targets_tensor.detach().cpu().tolist())

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(dataset)

        # Compute Rank Correlation Metrics across full dataset predictions
        if len(set(epoch_preds)) > 1 and len(set(epoch_targets)) > 1:
            spearman_rho, _ = spearmanr(epoch_preds, epoch_targets)
            pearson_r, _ = pearsonr(epoch_preds, epoch_targets)
        else:
            spearman_rho, pearson_r = 0.0, 0.0

        if np.isnan(spearman_rho): spearman_rho = 0.0
        if np.isnan(pearson_r): pearson_r = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        saved_flag = ""

        if spearman_rho > best_spearman_rho:
            best_spearman_rho = spearman_rho
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_egnn.pt")
            saved_flag = " ➔ [MODEL SAVED]"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Loss: {avg_epoch_loss:.4f} | Spearman ρ: {spearman_rho:.4f} | Pearson r: {pearson_r:.4f} | LR: {current_lr:.6f}{saved_flag}")

        if patience_counter >= patience:
            print("-" * 75)
            print(f"Early stopping triggered at Epoch {epoch}! Best Spearman ρ: {best_spearman_rho:.4f}")
            break

    print("-" * 75)
    print(f"Training complete. Best Spearman ρ achieved: {best_spearman_rho:.4f}")

if __name__ == "__main__":
    run_training()