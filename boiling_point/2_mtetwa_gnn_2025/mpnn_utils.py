import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
import pandas as pd
import numpy as np

def one_hot_encoding(x, permitted_list):
    """
    maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    tkes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return torch.tensor(atom_feature_vector, dtype=torch.float32)

def get_bond_pair(mol):
    '''
    constructs the edge_index tensor for a molecule
    '''
    bonds = mol.GetBonds()
    edge_index = [[], []]
    for bond in bonds:
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0].extend([i, j])    # forward direction
        edge_index[1].extend([j, i])    # reverse direction
    return edge_index

def bond_features(bond, 
                      use_stereochemistry = True):
    '''
    takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    '''
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return torch.tensor(bond_feature_vector, dtype=torch.float32)

def n_atom_features():
    '''
    determines number of atom features
    '''
    atom = Chem.MolFromSmiles('CC').GetAtomWithIdx(0)
    return len(atom_features(atom))

def n_bond_features():
    '''
    determines number of bond features
    '''
    bond = Chem.MolFromSmiles('CC').GetBondWithIdx(0)
    return len(bond_features(bond))

def mol_to_graph(smiles, y):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle invalid SMILES
        print(f"Invalid SMILES: {smiles}")
        return None  

    atom_feature_list = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_index_list = []
    edge_features_list = []

    for bond in mol.GetBonds():
        edge_index_list.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index_list.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])  # Symmetric
        edge_features_list.append(bond_features(bond))
        edge_features_list.append(bond_features(bond))

    # **Check for empty edge features**
    if not edge_features_list:
        print(f"Warning: No bonds found for SMILES {smiles}")
        return None  # Skip molecules with no bonds

    return Data(
        x=torch.stack(atom_feature_list),
        edge_index=torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_features_list),  # Ensure non-empty tensor
        y=torch.tensor([y], dtype=torch.float)
    )

class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_mlp(edge_attr)
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr

    def message(self, x_j, edge_attr):
        msg = torch.cat([x_j, edge_attr], dim=1)
        return self.node_mlp(msg)

    def update(self, aggr_out, x):
        return aggr_out + x

class MPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, output_dim):
        super().__init__()

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)

        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        for layer in self.mpnn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        x = global_add_pool(x, batch=data.batch)
        return self.readout(x)