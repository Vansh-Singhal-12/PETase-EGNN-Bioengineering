import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN

def run_training():
    print("Setting up training loop...")
    
    # 1. Load our dataset utility
    dataset = PETaseMutationDataset(pdb_path="data/6eqe.pdb")
    
    # 2. Initialize the EGNN model
    model = PETaseStabilityEGNN(num_amino_acids=20, emb_dim=32)
    
    # 3. Define our Loss Function and Optimizer
    # We use Mean Squared Error (MSE) to check our stability score accuracy
    criterion = nn.MSELoss()
    # Adam optimizer will handle tweaking the weights to lower our loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting a 5-epoch test training run...")
    print("-" * 40)
    
    # Put the model in training mode
    model.train()
    
    # Core training execution loop
    for epoch in range(1, 6):
        # Reset gradients so updates don't accumulate incorrectly
        optimizer.zero_grad()
        
        # Grab a sample graph, target score, and the specific mutation position
        graph_data, target_score, mutation_position = dataset[0]
        
        # Pass data through the model to get all 265 residue predictions
        all_predictions = model(graph_data)
        
       #Optimized: Force both tensors into matching flat 1D vector shapes [1]
        current_pred = all_predictions[mutation_position].view(-1)
        target_id = target_score.view(-1)

        # Calculate how far off our specific local prediction was from the target
        loss = criterion(current_pred, target_id)
        
        # Backpropagation: calculate the error gradients
        loss.backward()
        
        # Optimizer step: tweak the neural weights slightly
        optimizer.step()
        
        print(f"Epoch {epoch}/5 | Current Loss: {loss.item():.4f}")
        
    print("-" * 40)
    print("Training loop test finished successfully.")

if __name__ == "__main__":
    run_training()