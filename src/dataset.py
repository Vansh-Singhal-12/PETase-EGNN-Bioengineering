import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.data_loader import load_protein_as_graph

AA_PHYSICAL_PROPERTIES = {
    'A': [89.1,   1.8,  0.0, 0.0], 'R': [174.2, -4.5,  1.0, 4.0],
    'N': [132.1, -3.5,  0.0, 4.0], 'D': [133.1, -3.5, -1.0, 4.0],
    'C': [121.2,  2.5,  0.0, 1.0], 'Q': [146.2, -3.5,  0.0, 4.0],
    'E': [147.1, -3.5, -1.0, 4.0], 'G': [75.1,  -0.4,  0.0, 0.0],
    'H': [155.2, -3.2,  0.1, 2.0], 'I': [131.2,  4.5,  0.0, 0.0],
    'L': [131.2,  3.8,  0.0, 0.0], 'K': [146.2, -3.9,  1.0, 2.0],
    'M': [149.2,  1.9,  0.0, 0.0], 'F': [165.2,  2.8,  0.0, 0.0],
    'P': [115.1, -1.6,  0.0, 0.0], 'S': [105.1, -0.8,  0.0, 2.0],
    'T': [119.1, -0.7,  0.0, 2.0], 'W': [204.2, -0.9,  0.0, 2.0],
    'Y': [181.2, -1.3,  0.0, 3.0], 'V': [117.1,  4.2,  0.0, 0.0]
}

INT_TO_AA = {
    0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H',
    9: 'I', 10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T',
    17: 'W', 18: 'Y', 19: 'V'
}

class PETaseMutationDataset(Dataset):
    def __init__(self, pdb_path, csv_path=None, shield_radius=10.0):
        self.base_graph = load_protein_as_graph(pdb_path)
        self.shield_radius = shield_radius
        
        # Extract 3D C_alpha coordinates [num_nodes, 3]
        self.coords = self.base_graph.pos.numpy()
        num_nodes = self.coords.shape[0]
        
        # High-precision indices for IsPETase Catalytic Triad: S120, D177, H208
        # Clamp indices to valid graph bounds to prevent out-of-index errors
        self.triad_indices = [min(idx, num_nodes - 1) for idx in [120, 177, 208]]
        self.active_site_shield = self._compute_active_site_shield()
        
        # Convert integer amino acid IDs to continuous 4D biophysical property vectors
        wt_properties = []
        for i in range(num_nodes):
            aa_id = int(self.base_graph.x[i].item()) if self.base_graph.x[i].dim() == 0 else int(self.base_graph.x[i][0].item())
            aa_letter = INT_TO_AA.get(aa_id, 'A')
            wt_properties.append(AA_PHYSICAL_PROPERTIES[aa_letter])
            
        self.base_graph.x = torch.tensor(wt_properties, dtype=torch.float)
        
        if csv_path:
            print(f"Loading mutation dataset from: {csv_path}")
            self.mutations = pd.read_csv(csv_path)
        else:
            self.mutations = None

    def _compute_active_site_shield(self):
        """Calculates Euclidean distances to protect catalytic flexibility."""
        num_nodes = self.coords.shape[0]
        shield_mask = np.zeros(num_nodes, dtype=bool)
        
        triad_coords = self.coords[self.triad_indices]
        
        for i in range(num_nodes):
            # Distance from node i to all 3 catalytic triad residues
            distances = np.linalg.norm(triad_coords - self.coords[i], axis=1)
            if np.min(distances) < self.shield_radius:
                shield_mask[i] = True
                
        return torch.tensor(shield_mask, dtype=torch.bool)

    def __len__(self):
        if self.mutations is None:
            return 1
        return len(self.mutations)

    def __getitem__(self, idx):
        if self.mutations is None:
            return self.base_graph, torch.tensor([0.0]), torch.tensor([120]), self.active_site_shield
            
        row = self.mutations.iloc[idx]
        target_score = torch.tensor([row['stability_score']], dtype=torch.float)
        mutation_position = int(row['position_idx'])
        mutation_pos_tensor = torch.tensor([mutation_position], dtype=torch.long)
        
        mutant_aa_letter = str(row['mutation_type']).upper().strip()
        mutant_props = AA_PHYSICAL_PROPERTIES.get(mutant_aa_letter, [89.1, 1.8, 0.0, 0.0])
        
        mutated_graph = self.base_graph.clone()
        mutated_graph.x[mutation_position] = torch.tensor(mutant_props, dtype=torch.float)
        
        return mutated_graph, target_score, mutation_pos_tensor, self.active_site_shield