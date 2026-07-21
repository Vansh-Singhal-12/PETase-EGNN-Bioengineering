import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN

def run_training():
    print("Setting up training loop...")
    
    # 1. Load our dataset utility, pointing directly to our real CSV matrix
    dataset = PETaseMutationDataset(
        pdb_path="data/6eqe.pdb",
        csv_path="data/mutations.csv"
    )
    
    # 2. Initialize the EGNN model
    model = PETaseStabilityEGNN(num_amino_acids=20, emb_dim=32)
    
    # 3. Define our Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting training run over all {len(dataset)} dataset rows...")
    print("-" * 50)
    
    # Put the model in training mode
    model.train()
    
    # We will train for 5 complete passes (epochs) over the entire spreadsheet
    for epoch in range(1, 6):
        epoch_loss = 0.0
        
        # Inner loop: Iterate through every single mutation row in the CSV
        for idx in range(len(dataset)):
            # Reset gradients so updates don't accumulate incorrectly
            optimizer.zero_zero_grad() if hasattr(optimizer, 'zero_zero_grad') else optimizer.zero_grad()
            
            # Grab the specific graph, score, and position for this row
            graph_data, target_score, mutation_position = dataset[idx]
            
            # Pass data through the model to get all 265 residue predictions
            all_predictions = model(graph_data)
            
            # Force both tensors into matching flat 1D vector shapes [1]
            current_pred = all_predictions[mutation_position].view(-1)
            target_1d = target_score.view(-1)
            
            # Calculate how far off this specific variant prediction was
            loss = criterion(current_pred, target_1d)
            
            # Backpropagation: calculate the error gradients
            loss.backward()
            
            # Optimizer step: tweak the neural weights slightly
            optimizer.step()
            
            # Accumulate loss across the epoch
            epoch_loss += loss.item()
            
        # Compute the average loss across all rows for this epoch
        avg_epoch_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch}/5 | Average Dataset Loss: {avg_epoch_loss:.4f}")
        
    print("-" * 50)
    print("Training loop test finished successfully.")

if __name__ == "__main__":
    run_training()
