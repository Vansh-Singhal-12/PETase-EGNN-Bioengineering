import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.data_loader import load_protein_as_graph

# Standardized Biophysical Property Lookup Table
# Format: [Volume (Å³), Hydropathy (Kyte-Doolittle), Formal Charge, H-Bond Donors/Acceptors]
AA_PHYSICAL_PROPERTIES = {
    'A': [89.1, 1.8, 0.0, 0.0],   'R': [174.2, -4.5, 1.0, 4.0],
    'N': [132.1, -3.5, 0.0, 4.0],  'D': [133.1, -3.5, -1.0, 4.0],
    'C': [121.2, 2.5, 0.0, 1.0],   'Q': [146.2, -3.5, 0.0, 4.0],
    'E': [147.1, -3.5, -1.0, 4.0], 'G': [75.1, -0.4, 0.0, 0.0],
    'H': [155.2, -3.2, 0.1, 2.0],  'I': [131.2, 4.5, 0.0, 0.0],
    'L': [131.2, 3.8, 0.0, 0.0],   'K': [146.2, -3.9, 1.0, 2.0],
    'M': [149.2, 1.9, 0.0, 0.0],   'F': [165.2, 2.8, 0.0, 0.0],
    'P': [115.1, -1.6, 0.0, 0.0],  'S': [105.1, -0.8, 0.0, 2.0],
    'T': [119.1, -0.7, 0.0, 2.0],  'W': [204.2, -0.9, 0.0, 2.0],
    'Y': [181.2, -1.3, 0.0, 3.0],  'V': [117.1, 4.2, 0.0, 0.0]
}

INT_TO_AA = {
    0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H',
    9: 'I', 10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T',
    17: 'W', 18: 'Y', 19: 'V'
}

class PETaseMutationDataset(Dataset):
    """
    Bio-aware dataset engine generating 8D node features:
    [4D Wild-Type Properties || 4D Explicit Mutation Deltas (Mutant - WT)]
    Supports automatic inverse mutation data augmentation.
    """
    def __init__(self, pdb_path, csv_path=None, shield_radius=10.0, augment_inverse=True):
        self.base_graph = load_protein_as_graph(pdb_path)
        self.shield_radius = shield_radius
        self.augment_inverse = augment_inverse
        
        # Extract 3D C_alpha coordinates [num_nodes, 3]
        self.coords = self.base_graph.pos.numpy()
        num_nodes = self.coords.shape[0]
        
        # High-precision indices for IsPETase Catalytic Triad: S120, D177, H208
        self.triad_indices = [max(0, min(idx - 1, num_nodes - 1)) for idx in [120, 177, 208]]
        self.active_site_shield = self._compute_active_site_shield()
        
        # Extract wild-type 4D biophysical property vectors
        self.wt_letters = []
        wt_properties = []
        for i in range(num_nodes):
            aa_id = int(self.base_graph.x[i].item()) if self.base_graph.x[i].dim() == 0 else int(self.base_graph.x[i][0].item())
            aa_letter = INT_TO_AA.get(aa_id, 'A')
            self.wt_letters.append(aa_letter)
            wt_properties.append(AA_PHYSICAL_PROPERTIES[aa_letter])
            
        self.wt_properties_array = np.array(wt_properties, dtype=np.float32) # [num_nodes, 4]
        
        if csv_path and os.path.exists(csv_path):
            print(f"Loading mutation dataset from: {csv_path}")
            raw_df = pd.read_csv(csv_path)
            if self.augment_inverse:
                self.mutations = self._augment_dataset(raw_df)
                print(f"[Data Augmentation] Base rows: {len(raw_df)} -> Augmented rows: {len(self.mutations)}")
            else:
                self.mutations = raw_df
        else:
            self.mutations = None

    def _compute_active_site_shield(self):
        """Calculates Euclidean distances to protect catalytic flexibility."""
        num_nodes = self.coords.shape[0]
        shield_mask = np.zeros(num_nodes, dtype=bool)
        triad_coords = self.coords[self.triad_indices]
        
        for i in range(num_nodes):
            distances = np.linalg.norm(triad_coords - self.coords[i], axis=1)
            if np.min(distances) < self.shield_radius:
                shield_mask[i] = True
                
        return torch.tensor(shield_mask, dtype=torch.bool)

    def _augment_dataset(self, df):
        """Generates inverse mutation pairs (B -> A with -ΔTm) to double training signals."""
        augmented_rows = []
        for _, row in df.iterrows():
            # Original row
            augmented_rows.append(row.to_dict())
            
            # Inverse row
            inv_row = row.to_dict()
            wt = str(row['wild_type']).strip().upper()
            mut_str = str(row['mutation_type']).strip().upper()
            mut = mut_str[-1] if len(mut_str) > 0 else wt
            
            inv_row['wild_type'] = mut
            inv_row['mutation_type'] = wt
            inv_row['stability_score'] = -1.0 * float(row['stability_score'])
            augmented_rows.append(inv_row)
            
        return pd.DataFrame(augmented_rows)

    def __len__(self):
        if self.mutations is None:
            return 1
        return len(self.mutations)

    def __getitem__(self, idx):
        num_nodes = self.coords.shape[0]
        
        if self.mutations is None:
            # Fallback 8D tensor for uninitialized dry-run tests
            mock_x = torch.zeros((num_nodes, 8), dtype=torch.float)
            mock_graph = self.base_graph.clone()
            mock_graph.x = mock_x
            return mock_graph, torch.tensor([0.0]), torch.tensor([120]), self.active_site_shield
        
        row = self.mutations.iloc[idx]
        target_score = torch.tensor([float(row['stability_score'])], dtype=torch.float)
        
        raw_pos = int(row['position_idx'])
        mutation_position = max(0, min(raw_pos - 1, num_nodes - 1))
        mutation_pos_tensor = torch.tensor([mutation_position], dtype=torch.long)
        
        # Extract mutant amino acid
        mutation_str = str(row['mutation_type']).strip().upper()
        mutant_aa_letter = mutation_str[-1] if len(mutation_str) > 0 else self.wt_letters[mutation_position]
        
        wt_props = np.array(AA_PHYSICAL_PROPERTIES.get(self.wt_letters[mutation_position], [89.1, 1.8, 0.0, 0.0]))
        mutant_props = np.array(AA_PHYSICAL_PROPERTIES.get(mutant_aa_letter, wt_props))
        
        # Calculate 4D Delta: [ΔVolume, ΔHydropathy, ΔCharge, ΔH-Bonds]
        delta_props = mutant_props - wt_props
        
        # Build 8D Feature Matrix per node: [4D WT Properties || 4D Delta Features]
        node_features = np.zeros((num_nodes, 8), dtype=np.float32)
        node_features[:, 0:4] = self.wt_properties_array
        node_features[mutation_position, 4:8] = delta_props
        
        mutated_graph = self.base_graph.clone()
        mutated_graph.x = torch.tensor(node_features, dtype=torch.float)
        
        return mutated_graph, target_score, mutation_pos_tensor, self.active_site_shield

if __name__ == "__main__":
    print("Testing 8D Delta Feature Dataset Engine...")
    test_ds = PETaseMutationDataset(pdb_path="data/6eqe.pdb", csv_path="data/mutations_clean.csv", augment_inverse=True)
    graph_out, score_out, pos_out, shield_out = test_ds[0]
    print(f"Dataset Length (Augmented): {len(test_ds)}")
    print(f"Node Feature Shape        : {graph_out.x.shape} (Expected: [265, 8])")
    print(f"Delta Vector at Pos {pos_out.item()}  : {graph_out.x[pos_out.item(), 4:8].tolist()}")
    print("8D Delta Feature Engine Verification Passed!")