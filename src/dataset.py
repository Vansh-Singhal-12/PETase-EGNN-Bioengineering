import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from Bio.PDB import PDBParser

# =============================================================================
# IUPAC STANDARD AMINO ACID DICTIONARIES & BIOPHYSICAL PROPERTY MAPPINGS
# =============================================================================

# Standard 20 single-letter amino acid integer encoding mapping
AA_TO_INT = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7,
    'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
    'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

# 4D Biophysical Properties Table [Volume (Da), Hydropathy (Kyte-Doolittle), Formal Charge, H-Bond Capacity]
# Quantifies physical/chemical side-chain shock vectors upon mutation
AA_PROPERTIES = {
    'A': [89.1, 1.8, 0, 0],   'R': [174.2, -4.5, 1, 4], 'N': [132.1, -3.5, 0, 2],
    'D': [133.1, -3.5, -1, 2], 'C': [121.2, 2.5, 0, 0],  'E': [147.1, -3.5, -1, 2],
    'Q': [146.1, -3.5, 0, 2],  'G': [75.1, -0.4, 0, 0],  'H': [155.2, -3.2, 0.5, 2],
    'I': [131.2, 4.5, 0, 0],   'L': [131.2, 3.8, 0, 0],  'K': [146.2, -3.9, 1, 2],
    'M': [149.2, 1.9, 0, 0],   'F': [165.2, 2.8, 0, 0],  'P': [115.1, -1.6, 0, 0],
    'S': [105.1, -0.8, 0, 1],  'T': [119.1, -0.7, 0, 1], 'W': [204.2, -0.9, 0, 1],
    'Y': [181.2, -1.3, 0, 1],  'V': [117.1, 4.2, 0, 0]
}

# Z-score normalize 4D property vectors to zero mean and unit variance (mu=0, sigma=1)
props_matrix = np.array(list(AA_PROPERTIES.values()))
props_mean = props_matrix.mean(axis=0)
props_std = props_matrix.std(axis=0) + 1e-8
AA_PROPERTIES_NORM = {
    aa: ((np.array(props) - props_mean) / props_std).tolist()
    for aa, props in AA_PROPERTIES.items()
}


# =============================================================================
# PYTORCH GEOMETRIC DATASET CONTAINER FOR PETase MUTATIONS
# =============================================================================

class PETaseMutationDataset:
    def __init__(self, pdb_path="data/6eqe.pdb", csv_path="data/mutations_clean.csv", augment_inverse=True):
        """
        Initializes the PyG dataset wrapper for IsPETase mutation stability modeling.
        
        Args:
            pdb_path (str): Path to canonical 6EQE wild-type crystal structure PDB file.
            csv_path (str): Path to literature mutation tracking CSV file.
            augment_inverse (bool): If True, dynamically generates inverse mutations and synthetic multi-point pairs.
        """
        self.pdb_path = pdb_path
        self.csv_path = csv_path
        self.augment_inverse = augment_inverse
        
        # Load wild-type 3D spatial graph once into memory (265 C_alpha nodes, residues 29-293)
        self.base_graph = self._load_protein_as_graph(pdb_path)
        self.num_nodes = self.base_graph.x.size(0)  # 265 nodes
        
        self.df = pd.read_csv(csv_path)
        
        # Build training items array (single, inverse, and synthetic multi-point combinations)
        self.items = self._build_dataset_items()
        
    def _load_protein_as_graph(self, pdb_path):
        """
        Parses 3D Cartesian coordinates (x,y,z) of C_alpha backbone atoms from PDB structure
        and constructs thresholded Euclidean distance graph (d <= 8.0 Angstroms).
        """
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
        
        # Build thresholded Euclidean distance adjacency graph (d <= 8.0 Angstroms)
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        edge_indices = np.where((dist_matrix <= 8.0) & (dist_matrix > 0))
        # Fixed PyTorch warning by converting tuple of ndarrays to a single numpy array
        edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
        
        # Construct Z-score normalized 8D continuous node feature vectors [4D WT || 4D Delta=0]
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
        
        # 10.0 A Active Site Shield mask around catalytic triad.
        # CORRECTED: "Ser120, Asp177, His208" was wrong -- verified against the real
        # 6EQE sequence, those positions are Pro/Lys/Ile, not Ser/Asp/His at all.
        # The real catalytic triad (Austin et al. 2018 PNAS) is Ser160, Asp206, His237.
        # 0-indexed node coords (offset -29): Ser160->131, Asp206->177, His237->208
        catalytic_indices = [131, 177, 208]
        valid_cat_indices = [idx for idx in catalytic_indices if idx < num_nodes]
        
        cat_coords = coords[valid_cat_indices]
        dists_to_cat = np.linalg.norm(coords[:, None, :] - cat_coords[None, :, :], axis=-1)
        min_cat_dists = dists_to_cat.min(axis=-1)
        
        active_site_shield = torch.tensor(min_cat_dists <= 10.0, dtype=torch.bool)
        
        graph = Data(x=x_tensor, pos=pos_tensor, edge_index=edge_index)
        graph.active_site_shield = active_site_shield
        return graph

    def _map_pos_to_node_idx(self, p):
        """
        Unconditional node mapping: 6EQE resolved chain starts at residue 29 (node 0 = residue 29).
        Maps PDB residue numbers (29-293) to 0-indexed graph node indices (0-264).
        """
        p = int(p)
        if p >= 29:
            node_idx = p - 29
        else:
            node_idx = p
        return max(0, min(node_idx, self.num_nodes - 1))

    def _parse_pos_string(self, pos_str):
        """
        Parses single or comma-separated position strings into 0-indexed node coordinate lists.
        """
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

    def _parse_aa_string(self, aa_str):
        """
        Parses single or semicolon/comma-separated amino acid strings (e.g. 'I;S' or 'R;Q').
        """
        if not isinstance(aa_str, str):
            return [str(aa_str)[0] if str(aa_str) else 'A']
        parts = aa_str.replace(';', ',').split(',')
        cleaned = [p.strip()[0] for p in parts if p.strip()]
        return cleaned if cleaned else ['A']

    def _build_dataset_items(self):
        """
        Builds dataset rows: base single mutations, inverse mutations (B->A), and
        synthetic multi-point combinations (2-, 3-, 4-, 5-point) with synergy scaling.
        """
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
            # 1. Inverse single mutations (B -> A with inverted target -Delta T_m)
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

            # 3. Synthetic 3-point combinations (1.15x Synergy Multiplier)
            for i in range(num_raw):
                r1 = raw_rows[i]
                r2 = raw_rows[(i + 11) % num_raw]
                r3 = raw_rows[(i + 23) % num_raw]
                comb_pos = list(set(r1['pos_list'] + r2['pos_list'] + r3['pos_list']))
                comb_score = (r1['score'] + r2['score'] + r3['score']) * 1.15
                items.append((comb_pos, comb_score, r1['mut_type'], r1['wt_type'], False))

            # 4. Synthetic 4-point combinations (1.30x Synergy Multiplier)
            for i in range(num_raw):
                r1 = raw_rows[i]
                r2 = raw_rows[(i + 7) % num_raw]
                r3 = raw_rows[(i + 19) % num_raw]
                r4 = raw_rows[(i + 31) % num_raw]
                comb_pos = list(set(r1['pos_list'] + r2['pos_list'] + r3['pos_list'] + r4['pos_list']))
                comb_score = (r1['score'] + r2['score'] + r3['score'] + r4['score']) * 1.30
                items.append((comb_pos, comb_score, r1['mut_type'], r1['wt_type'], False))

            # 5. Synthetic 5-point combinations (1.40x Synergy Multiplier - matches HotPETase/FAST-PETase)
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
        """
        Retrieves a 3D spatial graph with injected 4D biophysical delta vectors at mutated node indices.
        """
        pos_list, score, mut_type, wt_type, is_inverse = self.items[idx]
        
        graph = self.base_graph.clone()
        
        # Parse single or multi-point amino acid codes
        m_codes = self._parse_aa_string(mut_type)
        w_codes = self._parse_aa_string(wt_type)
        
        num_nodes = graph.x.size(0)
        for i, p in enumerate(pos_list):
            if p < num_nodes:
                # Match corresponding AA code per position index
                m_c = m_codes[i] if i < len(m_codes) else m_codes[0]
                w_c = w_codes[i] if i < len(w_codes) else w_codes[0]
                
                m_props = np.array(AA_PROPERTIES_NORM.get(m_c, [0.0, 0.0, 0.0, 0.0]))
                w_props = np.array(AA_PROPERTIES_NORM.get(w_c, [0.0, 0.0, 0.0, 0.0]))
                delta_props = (m_props - w_props).tolist()
                
                # Inject 4D biophysical delta vector into nodes [channels 4:8]
                graph.x[p, 4:] = torch.tensor(delta_props, dtype=torch.float)
                
        target_tensor = torch.tensor([score], dtype=torch.float)
        mutation_pos_tensor = torch.tensor(pos_list, dtype=torch.long)
        shield_mask = graph.active_site_shield
        
        return graph, target_tensor, mutation_pos_tensor, shield_mask