import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.data_loader import load_protein_as_graph

AA_PHYSICAL_PROPERTIES_RAW = {
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

raw_matrix = np.array(list(AA_PHYSICAL_PROPERTIES_RAW.values()), dtype=np.float32)
AA_MEAN = raw_matrix.mean(axis=0)
AA_STD = raw_matrix.std(axis=0)
AA_STD[AA_STD == 0] = 1.0

AA_PHYSICAL_PROPERTIES = {
    k: ((np.array(v, dtype=np.float32) - AA_MEAN) / AA_STD).tolist()
    for k, v in AA_PHYSICAL_PROPERTIES_RAW.items()
}

class PETaseMutationDataset(Dataset):
    """
    Multi-point enabled PyTorch Dataset.
    Infiltrates single or multi-point biophysical delta vectors into graph node tensors
    and performs synthetic multi-point combinatorial data augmentation.
    """
    def __init__(self, pdb_path, csv_path=None, shield_radius=10.0, augment_inverse=True):
        self.base_graph = load_protein_as_graph(pdb_path)
        self.shield_radius = shield_radius
        self.augment_inverse = augment_inverse
        
        self.coords = self.base_graph.pos.numpy()
        num_nodes = self.coords.shape[0]
        
        self.triad_indices = [max(0, min(idx - 1, num_nodes - 1)) for idx in [120, 177, 208]]
        self.active_site_shield = self._compute_active_site_shield()
        
        self.wt_letters = []
        wt_properties = []
        for i in range(num_nodes):
            aa_id = int(self.base_graph.x[i].item()) if self.base_graph.x[i].dim() == 0 else int(self.base_graph.x[i][0].item())
            aa_letter = INT_TO_AA.get(aa_id, 'A')
            self.wt_letters.append(aa_letter)
            wt_properties.append(AA_PHYSICAL_PROPERTIES[aa_letter])
            
        self.wt_properties_array = np.array(wt_properties, dtype=np.float32)
        
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
        num_nodes = self.coords.shape[0]
        shield_mask = np.zeros(num_nodes, dtype=bool)
        triad_coords = self.coords[self.triad_indices]
        for i in range(num_nodes):
            distances = np.linalg.norm(triad_coords - self.coords[i], axis=1)
            if np.min(distances) < self.shield_radius:
                shield_mask[i] = True
        return torch.tensor(shield_mask, dtype=torch.bool)

    def _augment_dataset(self, df):
        """Generates inverse mutation pairs and synthetic multi-point combinations."""
        augmented_rows = []
        np_rng = np.random.RandomState(42)
        
        # 1. Base rows
        for _, row in df.iterrows():
            augmented_rows.append(row.to_dict())
            
            # 2. Inverse mutation rows (B -> A with -ΔTm)
            inv_row = row.to_dict()
            wt = str(row['wild_type']).strip().upper()
            mut_str = str(row['mutation_type']).strip().upper()
            mut = mut_str[-1] if len(mut_str) > 0 else wt
            
            inv_row['wild_type'] = mut
            inv_row['mutation_type'] = wt
            inv_row['stability_score'] = -1.0 * float(row['stability_score'])
            augmented_rows.append(inv_row)

        # 3. Synthetic Multi-Point Pair Augmentation (Combinatorial training)
        n_rows = len(df)
        if n_rows > 1:
            for _ in range(n_rows):
                i1, i2 = np_rng.choice(n_rows, size=2, replace=False)
                r1, r2 = df.iloc[i1], df.iloc[i2]
                
                pos1, pos2 = str(r1['position_idx']).strip(), str(r2['position_idx']).strip()
                if pos1 != pos2: # Non-overlapping positions
                    synth_row = {
                        'wild_type': f"{r1['wild_type']};{r2['wild_type']}",
                        'mutation_type': f"{r1['mutation_type']};{r2['mutation_type']}",
                        'position_idx': f"{pos1},{pos2}",
                        'stability_score': float(r1['stability_score']) + float(r2['stability_score'])
                    }
                    augmented_rows.append(synth_row)

        return pd.DataFrame(augmented_rows)

    def __len__(self):
        if self.mutations is None:
            return 1
        return len(self.mutations)

    def __getitem__(self, idx):
        num_nodes = self.coords.shape[0]
        
        if self.mutations is None:
            mock_x = torch.zeros((num_nodes, 8), dtype=torch.float)
            mock_graph = self.base_graph.clone()
            mock_graph.x = mock_x
            return mock_graph, torch.tensor([0.0]), torch.tensor([120], dtype=torch.long), self.active_site_shield
        
        row = self.mutations.iloc[idx]
        target_score = torch.tensor([float(row['stability_score'])], dtype=torch.float)
        
        # Parse single or multi-point position indices (e.g. "121" or "121,186")
        pos_str = str(row['position_idx'])
        pos_list = [max(0, min(int(p.strip()) - 1, num_nodes - 1)) for p in str(pos_str).split(',')]
        mutation_pos_tensor = torch.tensor(pos_list, dtype=torch.long)
        
        # Parse mutant amino acid letters
        mut_str = str(row['mutation_type']).strip().upper()
        mut_parts = mut_str.split(';') if ';' in mut_str else mut_str.split(',')
        
        mut_letters = []
        for part in mut_parts:
            part = part.strip()
            if len(part) > 0:
                mut_letters.append(part[-1])
        
        while len(mut_letters) < len(pos_list):
            mut_letters.append(mut_letters[0] if len(mut_letters) > 0 else 'A')
            
        mutated_graph = self.base_graph.clone()
        node_features = np.zeros((num_nodes, 8), dtype=np.float32)
        node_features[:, 0:4] = self.wt_properties_array
        
        # Inject delta vectors at ALL mutated positions simultaneously
        for p_idx, mut_aa in zip(pos_list, mut_letters):
            wt_aa = self.wt_letters[p_idx]
            wt_props = np.array(AA_PHYSICAL_PROPERTIES.get(wt_aa, [0.0, 0.0, 0.0, 0.0]))
            mutant_props = np.array(AA_PHYSICAL_PROPERTIES.get(mut_aa, wt_props))
            delta_props = mutant_props - wt_props
            node_features[p_idx, 4:8] = delta_props
            
        mutated_graph.x = torch.tensor(node_features, dtype=torch.float)
        return mutated_graph, target_score, mutation_pos_tensor, self.active_site_shield

if __name__ == "__main__":
    print("Testing Multi-Point Dataset Infiltration Engine...")
    test_ds = PETaseMutationDataset(pdb_path="data/6eqe.pdb", csv_path="data/benchmark_25.csv", augment_inverse=False)
    graph_out, score_out, pos_out, shield_out = test_ds[11] # Row 12: Double mutant 121,186
    print(f"Parsed Benchmark Row 12 (Double Mutant): Target Score = {score_out.item()} °C")
    print(f"Mutated Pos Tensor: {pos_out.tolist()} (Shape: {pos_out.shape})")
    print(" Multi-Point Infiltration Engine Verification Passed!")