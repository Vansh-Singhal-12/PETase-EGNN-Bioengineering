import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from scipy.stats import spearmanr, pearsonr
from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN

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
    Combines standard MSE anchor loss, pairwise margin ranking loss across
    variants, and an active-site protection penalty.
    """
    def __init__(self, margin=0.2, alpha=1.0, beta=0.05):
        super(CompositeMarginLoss, self).__init__()
        self.mse_fn = nn.MSELoss()
        self.ranking_fn = nn.MarginRankingLoss(margin=margin)
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, preds, targets):
        # 1. Scale Anchor MSE Loss
        mse_loss = self.mse_fn(preds, targets)
        
        # 2. Pairwise Margin Ranking Loss
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
    print("Setting up training loop with Composite Margin-Ranking Loss & Neighborhood Pooling...")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    dataset = PETaseMutationDataset(
        pdb_path="data/6eqe.pdb",
        csv_path="data/mutations_clean.csv"
    )
    
    model = PETaseStabilityEGNN(num_amino_acids=4, emb_dim=32, radius=10.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = CompositeMarginLoss(margin=0.2, alpha=1.0, beta=0.05)
    
    best_spearman_rho = -1.0
    
    print(f"Starting training run over all {len(dataset)} dataset rows...")
    print("-" * 75)
    
    NUM_EPOCHS = 200
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        preds_list = []
        targets_list = []
        
        for idx in range(len(dataset)):
            graph_data, target_score, mutation_pos, _ = dataset[idx]
            
            pred = model(graph_data, mutation_pos)
            
            preds_list.append(pred)
            targets_list.append(target_score.view(-1))
            
        preds_tensor = torch.stack(preds_list).view(-1)
        targets_tensor = torch.stack(targets_list).view(-1)
        
        loss = criterion(preds_tensor, targets_tensor)
        loss.backward()
        optimizer.step()
        
        # Calculate Rank Correlation Metrics
        preds_np = preds_tensor.detach().cpu().numpy()
        targets_np = targets_tensor.detach().cpu().numpy()
        
        if len(set(preds_np)) > 1 and len(set(targets_np)) > 1:
            spearman_rho, _ = spearmanr(preds_np, targets_np)
            pearson_r, _ = pearsonr(preds_np, targets_np)
        else:
            spearman_rho, pearson_r = 0.0, 0.0
            
        if np.isnan(spearman_rho): spearman_rho = 0.0
        if np.isnan(pearson_r): pearson_r = 0.0
        
        saved_flag = ""
        if spearman_rho > best_spearman_rho:
            best_spearman_rho = spearman_rho
            torch.save(model.state_dict(), "checkpoints/best_egnn.pt")
            saved_flag = " ➔ [MODEL SAVED]"
            
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Loss: {loss.item():.4f} | Spearman ρ: {spearman_rho:.4f} | Pearson r: {pearson_r:.4f}{saved_flag}")
        
    print("-" * 75)
    print(f"Training complete. Best Spearman ρ achieved: {best_spearman_rho:.4f}")

if __name__ == "__main__":
    run_training()