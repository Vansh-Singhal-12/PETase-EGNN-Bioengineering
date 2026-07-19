import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    """
    A single Equivariant Graph Neural Network Layer.
    Updates the amino acid features and their 3D coordinates together.
    """
    def __init__(self, emb_dim):
        super(EGNNLayer, self).__init__(aggr="sum")
        
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
        
        # Coordinate network: figures out the spatial shifts
        self.coord_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 1),
            nn.Tanh() # Keeps updates stable so things don't explode
        )

    def forward(self, h, pos, edge_index):
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_i, h_j, pos_i, pos_j):
        # Calculate squared distance between residue pairs
        # Distance squared is invariant (doesn't change if the protein rotates)
        coord_diff = pos_i - pos_j
        sq_dist = torch.sum(coord_diff ** 2, dim=-1, keepdim=True)
        
        # Combine node features and distance for the edge model
        edge_input = torch.cat([h_i, h_j, sq_dist], dim=-1)
        msg = self.edge_mlp(edge_input)
        
        # Scale the directional vector by our learned weights
        coord_weight = self.coord_mlp(msg)
        coord_msg = coord_diff * coord_weight
        
        return msg, coord_msg

    def aggregate(self, inputs, index, dim_size=None):
        node_msg, coord_msg = inputs
        
        # Sum up messages from all neighboring amino acids
        agg_node_msg = super().aggregate(node_msg, index, dim_size=dim_size)
        agg_coord_msg = super().aggregate(coord_msg, index, dim_size=dim_size)
        
        return agg_node_msg, agg_coord_msg

    def update(self, aggr_out, h, pos):
        agg_node_msg, agg_coord_msg = aggr_out
        
        # Generate new node features
        node_input = torch.cat([h, agg_node_msg], dim=-1)
        h_new = self.node_mlp(node_input)
        
        # Update the 3D coordinates based on neighbor vectors
        pos_new = pos + agg_coord_msg
        
        return h_new, pos_new


class PETaseStabilityEGNN(nn.Module):
    """
    Main model. Converts amino acid IDs to embeddings and passes them through EGNN layers.
    """
    def __init__(self, num_amino_acids=20, emb_dim=32):
        super(PETaseStabilityEGNN, self).__init__()
        
        # Maps integer values (0-19) to vector spaces
        self.embedding = nn.Embedding(num_amino_acids, emb_dim)
        
        # Two layers of message passing for structural depth
        self.egnn_layer1 = EGNNLayer(emb_dim)
        self.egnn_layer2 = EGNNLayer(emb_dim)
        
        # Final layer to output a single prediction score per residue
        self.regression_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, data):
        h = self.embedding(data.x)
        pos = data.pos
        edge_index = data.edge_index
        
        # Run through our equivariant blocks
        h, pos = self.egnn_layer1(h, pos, edge_index)
        h, pos = self.egnn_layer2(h, pos, edge_index)
        
        stability_predictions = self.regression_head(h)
        return stability_predictions

if __name__ == "__main__":
    print("Testing model compilation...")
    try:
        model = PETaseStabilityEGNN(num_amino_acids=20, emb_dim=32)
        print(model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Compilation error: {str(e)}")