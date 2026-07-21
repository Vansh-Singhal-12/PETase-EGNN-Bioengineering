import torch
import pandas as pd
from torch.utils.data import Dataset
from src.data_loader import load_protein_as_graph

class PETaseMutationDataset(Dataset):
    """
    Custom dataset to map structural 3D protein graphs to empirical mutation 
    spreadsheets, matching local sequence changes with stability metrics.
    """
    def __init__(self, pdb_path, csv_path=None):
        # Load the core 3D protein structural graph (wild-type backbone)
        self.base_graph = load_protein_as_graph(pdb_path)
        
        # Load empirical mutation data if a CSV path is provided
        if csv_path:
            print(f"Loading mutation dataset from: {csv_path}")
            self.mutations = pd.read_csv(csv_path)
        else:
            self.mutations = None

    def __len__(self):
        # Fall back to a test length of 1 if no spreadsheet is linked
        if self.mutations is None:
            return 1
        return len(self.mutations)

    def __getitem__(self, idx):
        # Default placeholder values for dry-run testing if no CSV exists
        if self.mutations is None:
            dummy_target = torch.tensor([0.5], dtype=torch.float)
            dummy_mutation_position = torch.tensor([120], dtype=torch.long)
            return self.base_graph, dummy_target, dummy_mutation_position
            
        # Extract specific variant row from the CSV matrix
        row = self.mutations.iloc[idx]
        
        # Parse the empirical target stability score (Delta-Delta G)
        target_score = torch.tensor([row['stability_score']], dtype=torch.float)
        
        # Pull the exact residue mutation position along the 265-node backbone
        mutation_position = torch.tensor([row['position_idx']], dtype=torch.long)
        
        return self.base_graph, target_score, mutation_position

if __name__ == "__main__":
    print("Testing dataset spreadsheet parser...")
    try:
        # Initialize dataset pointing directly to the new CSV matrix
        test_dataset = PETaseMutationDataset(
            pdb_path="data/6eqe.pdb", 
            csv_path="data/mutations.csv"
        )
        
        print("-" * 30)
        print(f"Total spreadsheet rows parsed: {len(test_dataset)}")
        
        # Sample index 0 (Row 1: Position 120, Score -1.25)
        graph_0, target_0, pos_0 = test_dataset[0]
        print(f"\n[Row 0] Target Position Index : {pos_0.item()}")
        print(f"[Row 0] True Stability Score   : {target_0.item():.2f}")
        
        # Sample index 1 (Row 2: Position 179, Score 2.40)
        graph_1, target_1, pos_1 = test_dataset[1]
        print(f"\n[Row 1] Target Position Index : {pos_1.item()}")
        print(f"[Row 1] True Stability Score   : {target_1.item():.2f}")
        print("-" * 30)
        print("Spreadsheet ingestion is working flawlessly.")
        
    except Exception as e:
        print(f"Error testing dataset spreadsheet script: {str(e)}")
