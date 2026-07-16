import torch
from torch.utils.data import Dataset
from src.data_loader import load_protein_as_graph

class PETaseMutationDataset(Dataset):
    """
    Custom dataset to pair our 3D protein graph with real mutation scores.
    """
    def __init__(self, pdb_path, mutation_df=None):
        # Load the core 3D protein structural graph
        self.base_graph = load_protein_as_graph(pdb_path)
        
        # I will connect the mutation spreadsheet entries here
        self.mutations = mutation_df

    def __len__(self):
        # If I don't have a spreadsheet yet, default to a dummy length of 1 for testing
        if self.mutations is None:
            return 1
        return len(self.mutations)

    def __getitem__(self, idx):
        # For now, I return our base graph and a placeholder target score for testing
        # Once I load a real CSV, this will pull the real stability values
        if self.mutations is None:
            dummy_target = torch.tensor([0.5], dtype=torch.float)
            return self.base_graph, dummy_target
            
        # This is where I will map the specific spreadsheet row to our graph
        row = self.mutations.iloc[idx]
        target_score = torch.tensor([row['stability_score']], dtype=torch.float)
        
        return self.base_graph, target_score

if __name__ == "__main__":
    print("Testing dataset utility script...")
    try:
        # Dry run test using our existing PDB file
        test_dataset = PETaseMutationDataset(pdb_path="data/6eqe.pdb")
        sample_graph, sample_target = test_dataset[0]
        
        print("-" * 30)
        print(f"Dataset items available : {len(test_dataset)}")
        print(f"Graph nodes verified    : {sample_graph.num_nodes}")
        print(f"Target tensor shape     : {sample_target.shape}")
        print("-" * 30)
        print("Dataset script is working smoothly.")
        
    except Exception as e:
        print(f"Error testing dataset script: {str(e)}")