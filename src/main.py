import torch
from src.data_loader import load_protein_as_graph
from src.model import PETaseStabilityEGNN

def run_pipeline():
    print("Running data-to-model pipeline...")
    
    pdb_path = "data/6eqe.pdb"
    
    # Load data using the graph converter
    protein_graph = load_protein_as_graph(pdb_path, distance_cutoff=8.0)
    
    # Initialize the EGNN model
    model = PETaseStabilityEGNN(num_amino_acids=20, emb_dim=32)
    model.eval()
    
    # Run the graph through the network layers
    with torch.no_grad():
        predictions = model(protein_graph)
        
    print("-" * 30)
    print(f"Output Tensor Shape : {predictions.shape}")
    print(f"Total Predicted Nodes: {predictions.size(0)}")
    print("-" * 30)
    
    print(f"Sample raw outputs (first 5 nodes):\n{predictions[:5].flatten()}")
    print("Everything is working smoothly.")

if __name__ == "__main__":
    run_pipeline()