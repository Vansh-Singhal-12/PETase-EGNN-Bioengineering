import torch
import pandas as pd
from torch.utils.data import Dataset
from src.data_loader import load_protein_as_graph

# Standardized Biophysical Property Lookup Table
# Format: [Molecular Weight (Da), Hydropathy Index, Formal Charge, H-Bond Capacity]
AA_PHYSICAL_PROPERTIES = {
    'A': [89.1,   1.8,  0.0, 0.0],  # Alanine
    'R': [174.2, -4.5,  1.0, 4.0],  # Arginine
    'N': [132.1, -3.5,  0.0, 4.0],  # Asparagine
    'D': [133.1, -3.5, -1.0, 4.0],  # Aspartate
    'C': [121.2,  2.5,  0.0, 1.0],  # Cysteine
    'Q': [146.2, -3.5,  0.0, 4.0],  # Glutamine
    'E': [147.1, -3.5, -1.0, 4.0],  # Glutamate
    'G': [75.1,  -0.4,  0.0, 0.0],  # Glycine
    'H': [155.2, -3.2,  0.1, 2.0],  # Histidine (partially positive at neutral pH)
    'I': [131.2,  4.5,  0.0, 0.0],  # Isoleucine
    'L': [131.2,  3.8,  0.0, 0.0],  # Leucine
    'K': [146.2, -3.9,  1.0, 2.0],  # Lysine
    'M': [149.2,  1.9,  0.0, 0.0],  # Methionine
    'F': [165.2,  2.8,  0.0, 0.0],  # Phenylalanine
    'P': [115.1, -1.6,  0.0, 0.0],  # Proline
    'S': [105.1, -0.8,  0.0, 2.0],  # Serine
    'T': [119.1, -0.7,  0.0, 2.0],  # Threonine
    'W': [204.2, -0.9,  0.0, 2.0],  # Tryptophan
    'Y': [181.2, -1.3,  0.0, 3.0],  # Tyrosine
    'V': [117.1,  4.2,  0.0, 0.0]   # Valine
}

# Reverse mapping for base graph construction (Int ID back to Letter)
INT_TO_AA = {
    0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H',
    9: 'I', 10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T',
    17: 'W', 18: 'Y', 19: 'V'
}

class PETaseMutationDataset(Dataset):
    """
    Bio-aware dataset engine that translates raw amino acid graphs into continuous 
    physical-chemical property matrices while executing coordinate-matched substitutions.
    """
    def __init__(self, pdb_path, csv_path=None):
        # Load standard wild-type graph
        self.base_graph = load_protein_as_graph(pdb_path)
        
        # Pre-convert the base graph's integer labels into real property vectors
        # Shape shifts from [num_nodes, 1] integer IDs to [num_nodes, 4] float features
        num_nodes = self.base_graph.x.shape[0]
        wt_properties = []
        for i in range(num_nodes):
            aa_id = int(self.base_graph.x[i].item())
            aa_letter = INT_TO_AA.get(aa_id, 'A')
            wt_properties.append(AA_PHYSICAL_PROPERTIES[aa_letter])
            
        # Store properties back into base graph features
        self.base_graph.x = torch.tensor(wt_properties, dtype=torch.float)
        
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
            dummy_target = torch.tensor([0.0], dtype=torch.float)
            dummy_pos = torch.tensor([120], dtype=torch.long)
            return self.base_graph, dummy_target, dummy_pos
            
        row = self.mutations.iloc[idx]
        target_score = torch.tensor([row['stability_score']], dtype=torch.float)
        mutation_position = int(row['position_idx'])
        mutation_pos_tensor = torch.tensor([mutation_position], dtype=torch.long)
        
        # Parse mutant identity
        mutant_aa_letter = str(row['mutation_type']).upper().strip()
        # Fallback to Alanine values if there's a character typo
        mutant_props = AA_PHYSICAL_PROPERTIES.get(mutant_aa_letter, [89.1, 1.8, 0.0, 0.0])
        
        # Clone graph to keep mutations isolated in memory
        mutated_graph = self.base_graph.clone()
        
        # Inject the continuous 4D property vector directly into the node feature row
        mutated_graph.x[mutation_position] = torch.tensor(mutant_props, dtype=torch.float)
        
        return mutated_graph, target_score, mutation_pos_tensor

if __name__ == "__main__":
    print("Testing continuous bio-feature pipeline...")
    test_ds = PETaseMutationDataset(pdb_path="data/6eqe.pdb", csv_path="data/mutations.csv")
    graph_out, score_out, pos_out = test_ds[0]
    print(f"Node feature tensor shape: {graph_out.x.shape} (Should be [265, 4])")
    print(f"Mutated node vector at position {pos_out.item()}: {graph_out.x[pos_out.item()].tolist()}")
