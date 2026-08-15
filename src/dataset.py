import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from Bio.PDB import PDBParser

# IUPAC standard 20 amino acid mapping
AA_TO_INT = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7,
    'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
    'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

# 4D Biophysical Properties Table [Volume (Da), Hydropathy, Charge, H-Bonds]
AA_PROPERTIES = {
    'A': [89.1, 1.8, 0, 0], 'R': [174.2, -4.5, 1, 4], 'N': [132.1, -3.5, 0, 2],
    'D': [133.1, -3.5, -1, 2], 'C': [121.2, 2.5, 0, 0], 'E': [147.1, -3.5, -1, 2],
    'Q': [146.1, -3.5, 0, 2], 'G': [75.1, -0.4, 0, 0], 'H': [155.2, -3.2, 0.5, 2],
    'I': [131.2, 4.5, 0, 0], 'L': [131.2, 3.8, 0, 0], 'K': [146.2, -3.9, 1, 2],
    'M': [149.2, 1.9, 0, 0], 'F': [165.2, 2.8, 0, 0], 'P': [115.1, -1.6, 0, 0],
    'S': [105.1, -0.8, 0, 1], 'T': [119.1, -0.7, 0, 1], 'W': [204.2, -0.9, 0, 1],
    'Y': [181.2, -1.3, 0, 1], 'V': [117.1, 4.2, 0, 0]
}

# Z-score normalize 4D property vectors
props_matrix = np.array(list(AA_PROPERTIES.values()))
props_mean = props_matrix.mean(axis=0)
props_std = props_matrix.std(axis=0) + 1e-8
AA_PROPERTIES_NORM = {
    aa: ((np.array(props) - props_mean) / props_std).tolist()
    for aa, props in AA_PROPERTIES.items()
}

class PETaseMutationDataset:
    def __init__(self, pdb_path="data/6eqe.pdb", csv_path="data/mutations_clean.csv", augment_inverse=True):
        self.pdb_path = pdb_path
        self.csv_path = csv_path
        self.augment_inverse = augment_inverse
        
        self.base_graph = self._load_protein_as_graph(pdb_path)
        self.num_nodes = self.base_graph.x.size(0)  # 265 nodes (residues 29-293)
        self.df = pd.read_csv(csv_path)
        
        self.items = self._build_dataset_items()
        
    def _load_protein_as_graph(self, pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("PETase", pdb_path)
        
        coords = []
        res_names = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id("CA"):
                        coords.append(residue["CA"].get_coord())
                        res_names.append(residue.get_resname())
                        
        coords = np.array(coords)
        num_nodes = len(coords)
        
        # Build Euclidean distance graph (cutoff d <= 8.0 A)
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        edge_indices = np.where((dist_matrix <= 8.0) & (dist_matrix > 0))
        edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
        
        # Construct Z-score normalized 8D continuous node features [4D WT || 4D Delta]
        x_feat = []
        for resname in res_names:
            try:
                from Bio.PDB.Polypeptide import three_to_one
                one_letter = three_to_one(resname)
            except:
                one_letter = 'A'
            wt_props = AA_PROPERTIES_NORM.get(one_letter, [0.0, 0.0, 0.0, 0.0])
            x_feat.append(wt_props + [0.0, 0.0, 0.0, 0.0])
            
        x_tensor = torch.tensor(x_feat, dtype=torch.float)
        pos_tensor = torch.tensor(coords, dtype=torch.float)
        
        # 10.0 A Active Site Shield mask around catalytic triad (Ser120, Asp177, His208)
        # In 0-indexed node coordinates: Ser120 = 120 - 29 = 91, Asp177 = 177 - 29 = 148, His208 = 208 - 29 = 179
        catalytic_indices = [91, 148, 179]
        valid_cat_indices = [idx for idx in catalytic_indices if idx < num_nodes]
        
        cat_coords = coords[valid_cat_indices]
        dists_to_cat = np.linalg.norm(coords[:, None, :] - cat_coords[None, :, :], axis=-1)
        min_cat_dists = dists_to_cat.min(axis=-1)
        
        active_site_shield = torch.tensor(min_cat_dists <= 10.0, dtype=torch.bool)
        
        graph = Data(x=x_tensor, pos=pos_tensor, edge_index=edge_index)
        graph.active_site_shield = active_site_shield
        return graph

    def _map_pos_to_node_idx(self, p):
        p = int(p)
        # UNCONDITIONAL OFFSET FIX: 6EQE resolved chain starts at residue 29 (node 0 = residue 29)
        if p >= 29:
            node_idx = p - 29
        else:
            node_idx = p
        return max(0, min(node_idx, self.num_nodes - 1))

    def _parse_pos_string(self, pos_str):
        if isinstance(pos_str, (int, np.integer)):
            raw_list = [int(pos_str)]
        elif isinstance(pos_str, str):
            parts = pos_str.replace('[', '').replace(']', '').split(',')
            raw_list = [int(p.strip()) for p in parts if p.strip()]
        elif isinstance(pos_str, (list, tuple)):
            raw_list = [int(p) for p in pos_str]
        else:
            raw_list = [0]
            
        return [self._map_pos_to_node_idx(p) for p in raw_list]

    def _build_dataset_items(self):
        items = []
        raw_rows = []
        
        for idx, row in self.df.iterrows():
            pos_list = self._parse_pos_string(row['position_idx'])
            score = float(row['stability_score'])
            mut_type = str(row.get('mutation_type', 'A'))
            wt_type = str(row.get('wild_type', 'A'))
            
            raw_rows.append({
                'pos_list': pos_list,
                'score': score,
                'mut_type': mut_type,
                'wt_type': wt_type
            })
            items.append((pos_list, score, mut_type, wt_type, False))
            
        if self.augment_inverse:
            # 1. Inverse single mutations (B -> A with -Delta T_m)
            for r in raw_rows:
                items.append((r['pos_list'], -r['score'], r['wt_type'], r['mut_type'], True))
                
            num_raw = len(raw_rows)
            # 2. Synthetic 2-point combinations
            for i in range(num_raw):
                r1 = raw_rows[i]
                r2 = raw_rows[(i + 17) % num_raw]
                comb_pos = list(set(r1['pos_list'] + r2['pos_list']))
                comb_score = r1['score'] + r2['score']
                items.append((comb_pos, comb_score, r1['mut_type'], r1['wt_type'], False))

            # 3. Synthetic 3-point combinations (1.15x Synergy)
            for i in range(num_raw):
                r1 = raw_rows[i]
                r2 = raw_rows[(i + 11) % num_raw]
                r3 = raw_rows[(i + 23) % num_raw]
                comb_pos = list(set(r1['pos_list'] + r2['pos_list'] + r3['pos_list']))
                comb_score = (r1['score'] + r2['score'] + r3['score']) * 1.15
                items.append((comb_pos, comb_score, r1['mut_type'], r1['wt_type'], False))

            # 4. Synthetic 4-point combinations (1.30x Synergy)
            for i in range(num_raw):
                r1 = raw_rows[i]
                r2 = raw_rows[(i + 7) % num_raw]
                r3 = raw_rows[(i + 19) % num_raw]
                r4 = raw_rows[(i + 31) % num_raw]
                comb_pos = list(set(r1['pos_list'] + r2['pos_list'] + r3['pos_list'] + r4['pos_list']))
                comb_score = (r1['score'] + r2['score'] + r3['score'] + r4['score']) * 1.30
                items.append((comb_pos, comb_score, r1['mut_type'], r1['wt_type'], False))

            # 5. Synthetic 5-point combinations (1.40x Synergy)
            for i in range(num_raw):
                r1 = raw_rows[i]
                r2 = raw_rows[(i + 5) % num_raw]
                r3 = raw_rows[(i + 13) % num_raw]
                r4 = raw_rows[(i + 29) % num_raw]
                r5 = raw_rows[(i + 37) % num_raw]
                comb_pos = list(set(r1['pos_list'] + r2['pos_list'] + r3['pos_list'] + r4['pos_list'] + r5['pos_list']))
                comb_score = (r1['score'] + r2['score'] + r3['score'] + r4['score'] + r5['score']) * 1.40
                items.append((comb_pos, comb_score, r1['mut_type'], r1['wt_type'], False))

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pos_list, score, mut_type, wt_type, is_inverse = self.items[idx]
        
        graph = self.base_graph.clone()
        
        m_code = mut_type[0] if len(mut_type) > 0 else 'A'
        w_code = wt_type[0] if len(wt_type) > 0 else 'A'
            
        m_props = np.array(AA_PROPERTIES_NORM.get(m_code, [0.0, 0.0, 0.0, 0.0]))
        w_props = np.array(AA_PROPERTIES_NORM.get(w_code, [0.0, 0.0, 0.0, 0.0]))
        
        delta_props = (m_props - w_props).tolist()
        
        num_nodes = graph.x.size(0)
        for p in pos_list:
            if p < num_nodes:
                graph.x[p, 4:] = torch.tensor(delta_props, dtype=torch.float)
                
        target_tensor = torch.tensor([score], dtype=torch.float)
        mutation_pos_tensor = torch.tensor(pos_list, dtype=torch.long)
        shield_mask = graph.active_site_shield
        
        return graph, target_tensor, mutation_pos_tensor, shield_mask