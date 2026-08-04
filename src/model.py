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
        
        # Node network: updates main amino acid features
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
            nn.Tanh() # Prevents spatial coordinate explosions
        )

    def forward(self, h, pos, edge_index):
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_i, h_j, pos_i, pos_j):
        coord_diff = pos_i - pos_j
        sq_dist = torch.sum(coord_diff ** 2, dim=-1, keepdim=True)
        
        edge_input = torch.cat([h_i, h_j, sq_dist], dim=-1)
        msg = self.edge_mlp(edge_input)
        
        coord_weight = self.coord_mlp(msg)
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
        
        # Residual skip connection for feature stability
        h_new = h + self.node_mlp(node_input)
        pos_new = pos + agg_coord_msg
        
        return h_new, pos_new


class PETaseStabilityEGNN(nn.Module):
    """
    Main EGNN model featuring Undiluted Mutation Direct Infiltration + 10Å Spatial Neighborhood Context.
    """
    def __init__(self, num_amino_acids=8, emb_dim=32, radius=10.0):
        super(PETaseStabilityEGNN, self).__init__()
        self.radius = radius
        
        # Linear projection layer for 8D continuous biophysical features
        self.embedding = nn.Linear(num_amino_acids, emb_dim)
        
        # Dual-layer message passing
        self.egnn_layer1 = EGNNLayer(emb_dim)
        self.egnn_layer2 = EGNNLayer(emb_dim)
        
        # Regression head taking [Mutated_Node_Embedding (32D) || Pooled_Neighborhood_Embedding (32D)] = 64D
        self.regression_head = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, data, mutation_pos):
        h = self.embedding(data.x.float()) # [N, 32]
        pos = data.pos                     # [N, 3]
        edge_index = data.edge_index       # [2, E]
        
        # Run through E(3)-equivariant message-passing layers
        h, pos = self.egnn_layer1(h, pos, edge_index)
        h, pos = self.egnn_layer2(h, pos, edge_index)
        
        # 1. Isolated Mutated Node Embedding (100% Full Strength Signal) -> Flattened to [32]
        mutated_node_h = h[mutation_pos].view(-1) # [32]
        
        # 2. Smooth Gaussian 10Å Neighborhood Context -> Flattened to [32]
        mut_coord = pos[mutation_pos].view(-1) # [3]
        distances = torch.norm(pos - mut_coord, dim=-1)
        weights = torch.exp(-0.5 * (distances / self.radius) ** 2).unsqueeze(-1) # [N, 1]
        pooled_h = ((h * weights).sum(dim=0) / (weights.sum(dim=0) + 1e-8)).view(-1) # [32]
        
        # 3. Concatenate Mutation Signal + Spatial Context (both 1D tensors -> [64])
        combined_representation = torch.cat([mutated_node_h, pooled_h], dim=-1) # [64]
        
        out = self.regression_head(combined_representation)
        return out.squeeze(-1)


if __name__ == "__main__":
    print("Testing Undiluted EGNN Model Compilation...")
    try:
        from torch_geometric.data import Data
        mock_data = Data(
            x=torch.randn(265, 8),
            pos=torch.randn(265, 3),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        )
        model = PETaseStabilityEGNN(num_amino_acids=8, emb_dim=32, radius=10.0)
        out = model(mock_data, torch.tensor([120], dtype=torch.long))
        print(model)
        print(f"\nTest prediction output shape: {out.shape}")
        print("\n Model compiled and passed test forward pass with zero shape errors!")
    except Exception as e:
        print(f"\n Compilation Error: {str(e)}")