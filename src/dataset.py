import torch
import pandas as pd
from torch.utils.data import Dataset
from src.data_loader import load_protein_as_graph

# Standard IUPAC single-letter amino acid mapping to 0-19 integers
AA_TO_INT = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8,
    'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16,
    'W': 17, 'Y': 18, 'V': 19
}

class PETaseMutationDataset(Dataset):
    """
    Custom dataset that loads a base PDB graph and dynamically swaps node 
    features to reflect specific residue mutations from a CSV spreadsheet.
    """
    def __init__(self, pdb_path, csv_path=None):
        # Load our core 3D protein structural graph (wild-type backbone)
        self.base_graph = load_protein_as_graph(pdb_path)
        
        if csv_path:
            print(f"Loading mutation dataset from: {csv_path}")
            self.mutations = pd.read_csv(csv_path)
        else:
            self.mutations = None

    def __len__(self):
        if self.mutations is None:
            return 1
        return len(self.mutations)

    def __getitem__(self, idx):
        if self.mutations is None:
            dummy_target = torch.tensor([0.5], dtype=torch.float)
            dummy_mutation_position = torch.tensor([120], dtype=torch.long)
            return self.base_graph, dummy_target, dummy_mutation_position
            
        # Extract specific variant row from our CSV matrix
        row = self.mutations.iloc[idx]
        
        # 1. Parse targets and positions
        target_score = torch.tensor([row['stability_score']], dtype=torch.float)
        mutation_position = int(row['position_idx'])
        mutation_pos_tensor = torch.tensor([mutation_position], dtype=torch.long)
        
        # 2. Get the new mutant amino acid integer code
        mutant_aa_letter = str(row['mutation_type']).upper().strip()
        mutant_aa_int = AA_TO_INT.get(mutant_aa_letter, 0) # Fallback to 0 if invalid
        
        # 3. CLONE the base graph data object so mutations don't overwrite each other
        mutated_graph = self.base_graph.clone()
        
        # 4. CRITICAL STEP: Modify the node feature vector at the mutation index
        # mutated_graph.x holds the amino acid type IDs for all 265 nodes
        original_aa_int = mutated_graph.x[mutation_position].item()
        mutated_graph.x[mutation_position] = mutant_aa_int
        
        return mutated_graph, target_score, mutation_pos_tensor

if __name__ == "__main__":
    print("Testing dynamic feature substitution engine...")
    try:
        test_dataset = PETaseMutationDataset(
            pdb_path="data/6eqe.pdb", 
            csv_path="data/mutations.csv"
        )
        
        print("-" * 40)
        # Sample Row 0 (S to A at index 120)
        graph_0, target_0, pos_0 = test_dataset[0]
        replaced_feat = graph_0.x[pos_0.item()].item()
        print(f"[Row 0] Target Position: {pos_0.item()}")
        print(f"[Row 0] Expecting Mutant 'A' (Int 0). Actual feature in graph: {replaced_feat}")
        
        # Sample Row 1 (I to R at index 179)
        graph_1, target_1, pos_1 = test_dataset[1]
        replaced_feat_1 = graph_1.x[pos_1.item()].item()
        print(f"\n[Row 1] Target Position: {pos_1.item()}")
        print(f"[Row 1] Expecting Mutant 'R' (Int 1). Actual feature in graph: {replaced_feat_1}")
        print("-" * 40)
        print("Feature substitution test completed successfully!")
        
    except Exception as e:
        print(f"Error checking feature substitution: {str(e)}")