import torch
from gcn_utils import GCNModel  
from mpnn_utils import MPNN
from mpnn_utils import n_atom_features, n_bond_features

def load_gcn_model(path='gcn_model.pth', device='cpu'):
    model = GCNModel(num_features=79, hidden_size=32, num_classes=1)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)



def load_mpnn_model(path='mpnn_model.pth', device='cpu'):
    model = MPNN(
        node_dim=n_atom_features(),    
        edge_dim=n_bond_features(),
        hidden_dim=64,               
        num_layers=3,
        output_dim=1
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)

