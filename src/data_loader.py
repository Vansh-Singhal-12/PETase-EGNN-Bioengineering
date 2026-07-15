import os
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser

# Mapping the 20 standard amino acids to numbers so the computer can read them
AMINO_ACID_MAPPING = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

def load_protein_as_graph(pdb_path, distance_cutoff=8.0):
    """
    Reads a PDB file, finds the Alpha Carbons, and builds a 3D graph.
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Could not find PDB file at: {pdb_path}")
        
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    coordinates = []
    node_features = []
    
    # Loop through the PDB file structure to extract atoms
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip water molecules and non-protein atoms
                if residue.get_id()[0] != " ":
                    continue
                
                # Make sure it's one of the standard 20 amino acids
                res_name = residue.get_resname()
                if res_name not in AMINO_ACID_MAPPING:
                    continue
                
                # Grab the Alpha Carbon (CA) to represent the residue position
                if "CA" in residue:
                    ca_atom = residue["CA"]
                    coordinates.append(ca_atom.get_coord())
                    node_features.append(AMINO_ACID_MAPPING[res_name])
                    
    # Convert our lists into PyTorch tensors
    positions = torch.tensor(coordinates, dtype=torch.float) 
    x_features = torch.tensor(node_features, dtype=torch.long) 
    
    num_nodes = positions.size(0)
    edge_start = []
    edge_end = []
    
    # Calculate distances between residues to create connections (edges)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Euclidean distance formula
                distance = torch.norm(positions[i] - positions[j])
                if distance <= distance_cutoff:
                    edge_start.append(i)
                    edge_end.append(j)
                    
    edge_index = torch.tensor([edge_start, edge_end], dtype=torch.long) 
    
    # Package everything into a PyTorch Geometric Data object
    protein_graph = Data(
        x=x_features,          
        pos=positions,         
        edge_index=edge_index  
    )
    
    return protein_graph

if __name__ == "__main__":
    target_pdb = "data/6eqe.pdb"
    
    print(f"Parsing structure file: {target_pdb}")
    try:
        graph_data = load_protein_as_graph(target_pdb, distance_cutoff=8.0)
        
        print("Graph compiled successfully.")
        print("-" * 30)
        print(f"Amino Acid Nodes (CA atoms): {graph_data.num_nodes}")
        print(f"Spatial Connections (Edges): {graph_data.num_edges}")
        print(f"Node Tensor Shape          : {graph_data.x.shape}")
        print(f"Coordinate Tensor Shape    : {graph_data.pos.shape}")
        print("-" * 30)
        
    except Exception as error:
        print(f"Error parsing file: {str(error)}")