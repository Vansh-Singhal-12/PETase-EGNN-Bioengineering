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


def custom_active_site_loss(all_predictions, target_score, mutation_position, shield_mask, lambda_shield=0.1):
    """
    Combines primary mutation target MSE loss with an active-site penalty term.
    Penalizes predicted destabilization (negative values) on active-site shielded residues.
    """
    # 1. Primary MSE Loss at the mutation position
    pred_mutation = all_predictions[mutation_position].view(-1)
    target_1d = target_score.view(-1)
    mse_loss = nn.MSELoss()(pred_mutation, target_1d)
    
    # 2. Active-Site Shield Penalty:
    # ReLU(-x) activates whenever predicted stability shift inside the shield drops below 0.0
    shielded_preds = all_predictions[shield_mask]
    shield_destabilization_penalty = torch.mean(torch.relu(-shielded_preds))
    
    # Combined Loss
    total_loss = mse_loss + (lambda_shield * shield_destabilization_penalty)
    return total_loss, mse_loss.item()


def run_training():
    print("Setting up training loop with Custom Active-Site Loss & Checkpointing...")
    
    # Ensure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)
    
    dataset = PETaseMutationDataset(
        pdb_path="data/6eqe.pdb",
        csv_path="data/mutations.csv"
    )
    
    model = PETaseStabilityEGNN(num_amino_acids=4, emb_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_spearman_rho = -1.0  # Track highest Spearman correlation for checkpointing
    
    print(f"Starting training run over all {len(dataset)} dataset rows...")
    print("-" * 75)
    
    model.train()
    
    for epoch in range(1, 6):
        epoch_loss = 0.0
        predictions_list = []
        targets_list = []
        
        for idx in range(len(dataset)):
            optimizer.zero_grad()
            
            graph_data, target_score, mutation_position, active_site_shield = dataset[idx]
            
            # Forward pass through EGNN
            all_predictions = model(graph_data)
            
            # Compute Custom Shield Loss
            loss, base_mse = custom_active_site_loss(
                all_predictions, target_score, mutation_position, active_site_shield, lambda_shield=0.05
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Store predictions for rank evaluation
            predictions_list.append(all_predictions[mutation_position].item())
            targets_list.append(target_score.item())
            
        avg_epoch_loss = epoch_loss / len(dataset)
        
        # Compute Rank Correlation Metrics
        if len(set(predictions_list)) > 1 and len(set(targets_list)) > 1:
            spearman_rho, _ = spearmanr(predictions_list, targets_list)
            pearson_r, _ = pearsonr(predictions_list, targets_list)
        else:
            spearman_rho, pearson_r = 0.0, 0.0
            
        # Fallback if result is NaN
        if np.isnan(spearman_rho): spearman_rho = 0.0
        if np.isnan(pearson_r): pearson_r = 0.0
        
        # Checkpoint Saver: Save model weights whenever Spearman rho improves
        saved_flag = ""
        if spearman_rho > best_spearman_rho:
            best_spearman_rho = spearman_rho
            torch.save(model.state_dict(), "checkpoints/best_egnn.pt")
            saved_flag = " ➔ [MODEL SAVED]"
            
        print(f"Epoch {epoch}/5 | Loss: {avg_epoch_loss:.4f} | Spearman ρ: {spearman_rho:.4f} | Pearson r: {pearson_r:.4f}{saved_flag}")
        
    print("-" * 75)
    print(f"Training complete. Best Spearman ρ achieved: {best_spearman_rho:.4f}")

if __name__ == "__main__":
    run_training()