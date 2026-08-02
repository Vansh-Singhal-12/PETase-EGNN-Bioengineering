import torch
from src.data_loader import load_protein_as_graph
from src.model import PETaseStabilityEGNN

def run_pipeline():
    print("Running data-to-model pipeline check...")
    
    pdb_path = "data/6eqe.pdb"
    
    # Load data using the graph converter
    protein_graph = load_protein_as_graph(pdb_path, distance_cutoff=8.0)
    
    # Initialize the EGNN model
    model = PETaseStabilityEGNN(num_amino_acids=20, emb_dim=32, radius=10.0)
    model.eval()
    
    # Pick a dummy mutation position (e.g., residue index 0) for the dry run
    dummy_mutation_pos = 0
    
    # Run the graph through the network layers
    with torch.no_grad():
        prediction = model(protein_graph, dummy_mutation_pos)
        
    print("-" * 40)
    print(f"Output Tensor: {prediction}")
    print(f"Output Shape : {prediction.shape}")
    print("-" * 40)
    print("Pipeline check passed smoothly.")

if __name__ == "__main__":
    run_pipeline()