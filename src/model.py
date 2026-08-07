import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    """
    Stabilized Equivariant Graph Neural Network Layer.
    Updates amino acid features and 3D coordinates with damped spatial shifts.
    """
    def __init__(self, emb_dim=32, coord_scale=0.1):
        super(EGNNLayer, self).__init__(aggr='add')
        self.coord_scale = coord_scale
        self.edge_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 1),
            nn.Tanh()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, h, pos, edge_index):
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_i, h_j, pos_i, pos_j):
        rel_pos = pos_i - pos_j
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)
        edge_feat = torch.cat([h_i, h_j, dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_feat)
        return m_ij

    def update(self, aggr_out, h, pos, edge_index):
        # Invariant Node State Update with Residual Skip Connection
        h_new = h + self.node_mlp(torch.cat([h, aggr_out], dim=-1))
        
        # Equivariant Spatial Coordinate Update with Smooth Gaussian RBF Weighting
        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)
        m_ij = self.edge_mlp(torch.cat([h[row], h[col], dist_sq], dim=-1))
        coord_weights = self.coord_mlp(m_ij)
        
        rbf_weight = torch.exp(-0.5 * (torch.sqrt(dist_sq + 1e-8) / 10.0) ** 2)
        delta_pos = torch.zeros_like(pos)
        delta_pos.index_add_(0, row, rel_pos * coord_weights * rbf_weight * self.coord_scale)
        
        pos_new = pos + delta_pos
        return h_new, pos_new


class PETaseStabilityEGNN(nn.Module):
    def __init__(self, in_dim=8, emb_dim=32, dropout=0.2):
        super(PETaseStabilityEGNN, self).__init__()
        self.embedding = nn.Linear(in_dim, emb_dim)
        self.layer1 = EGNNLayer(emb_dim=emb_dim, coord_scale=0.1)
        self.layer2 = EGNNLayer(emb_dim=emb_dim, coord_scale=0.1)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Node-level readout head for Active Site Shield destabilization tracking
        self.node_readout = nn.Linear(emb_dim, 1)
        
        # 64D dual-tensor regression head (32D sum-pooled mutated node + 32D 10A spatial context)
        self.regression_head = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, graph_data, mutation_pos):
        h = self.embedding(graph_data.x.float())
        pos = graph_data.pos.float()
        edge_index = graph_data.edge_index

        h, pos = self.layer1(h, pos, edge_index)
        h, pos = self.layer2(h, pos, edge_index)

        # Per-node stability predictions for Active Site Shield loss evaluation
        node_preds = self.node_readout(h).view(-1)

        # Handle single-point or multi-point mutation position inputs
        if isinstance(mutation_pos, (list, tuple, torch.Tensor)):
            pos_list = torch.tensor(mutation_pos, dtype=torch.long, device=h.device) if not isinstance(mutation_pos, torch.Tensor) else mutation_pos.long()
        else:
            pos_list = torch.tensor([mutation_pos], dtype=torch.long, device=h.device)

        # SUM-POOLED FEATURE VECTOR: Accumulates multi-point signals cleanly
        mutated_node_h = h[pos_list].sum(dim=0).unsqueeze(0)  # Shape: [1, 32]

        # 10A Gaussian RBF spatial neighborhood pooling
        mutated_coords = pos[pos_list]
        dist_matrix = torch.cdist(pos, mutated_coords)
        min_dists, _ = torch.min(dist_matrix, dim=-1)
        
        rbf_weights = torch.exp(-0.5 * (min_dists / 10.0) ** 2).unsqueeze(-1)
        pooled_h = (h * rbf_weights).sum(dim=0, keepdim=True) / (rbf_weights.sum() + 1e-8)

        combined_h = torch.cat([mutated_node_h, pooled_h], dim=-1)  # Shape: [1, 64]
        combined_h = self.dropout(combined_h)

        prediction = self.regression_head(combined_h)
        return prediction.view(-1), node_preds