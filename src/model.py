import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    """
    Stabilized Equivariant Graph Neural Network Layer.
    Updates amino acid features and 3D coordinates with damped spatial shifts.
    """
    def __init__(self, emb_dim, coord_scale=0.1):
        super(EGNNLayer, self).__init__(aggr="sum")
        self.coord_scale = coord_scale
        
        # Edge network: handles neighbor features and distance
        self.edge_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # Node network: updates the main amino acid features
        self.node_mlp = nn.Sequential(
            nn.Linear(emb_dim + emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # Coordinate network: figures out spatial shifts
        self.coord_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 1),
            nn.Tanh() # Stable updates
        )

    def forward(self, h, pos, edge_index):
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_i, h_j, pos_i, pos_j):
        coord_diff = pos_i - pos_j
        sq_dist = torch.sum(coord_diff ** 2, dim=-1, keepdim=True)
        
        edge_input = torch.cat([h_i, h_j, sq_dist], dim=-1)
        msg = self.edge_mlp(edge_input)
        
        coord_weight = self.coord_mlp(msg)
        # Apply scaling factor to prevent coordinate runaway
        coord_msg = coord_diff * coord_weight * self.coord_scale
        
        return msg, coord_msg

    def aggregate(self, inputs, index, dim_size=None):
        node_msg, coord_msg = inputs
        
        agg_node_msg = super().aggregate(node_msg, index, dim_size=dim_size)
        agg_coord_msg = super().aggregate(coord_msg, index, dim_size=dim_size)
        
        return agg_node_msg, agg_coord_msg

    def update(self, aggr_out, h, pos):
        agg_node_msg, agg_coord_msg = aggr_out
        
        node_input = torch.cat([h, agg_node_msg], dim=-1)
        # Residual connection on node features for training stability
        h_new = h + self.node_mlp(node_input)
        
        pos_new = pos + agg_coord_msg
        
        return h_new, pos_new


class PETaseStabilityEGNN(nn.Module):
    """
    Main model utilizing EGNN layers with smooth spatial neighborhood pooling.
    """
    def __init__(self, num_amino_acids=20, emb_dim=32, radius=10.0):
        super(PETaseStabilityEGNN, self).__init__()
        self.radius = radius
        
        self.embedding = nn.Linear(num_amino_acids, emb_dim)
        
        self.egnn_layer1 = EGNNLayer(emb_dim)
        self.egnn_layer2 = EGNNLayer(emb_dim)
        
        self.regression_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, data, mutation_pos):
        h = self.embedding(data.x.float())
        pos = data.pos
        edge_index = data.edge_index
        
        # Run through equivariant blocks
        h, pos = self.egnn_layer1(h, pos, edge_index)
        h, pos = self.egnn_layer2(h, pos, edge_index)
        
        # --- Smooth Gaussian Neighborhood Weighting ---
        mut_coord = pos[mutation_pos] # [3]
        distances = torch.norm(pos - mut_coord, dim=-1) # [N]
        
        # Soft continuous weights decaying smooth toward 0 past radius
        weights = torch.exp(-0.5 * (distances / self.radius) ** 2).unsqueeze(-1) # [N, 1]
        
        # Weighted mean aggregation prevents hard threshold step-discontinuities
        pooled_h = (h * weights).sum(dim=0) / (weights.sum(dim=0) + 1e-8)
            
        out = self.regression_head(pooled_h)
        return out.squeeze(-1)

if __name__ == "__main__":
    print("Testing model compilation...")
    try:
        model = PETaseStabilityEGNN(num_amino_acids=4, emb_dim=32, radius=10.0)
        print(model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Compilation error: {str(e)}")