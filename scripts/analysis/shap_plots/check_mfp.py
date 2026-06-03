import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

data = pd.read_csv('data/database/merged_CN_YSI_SMILES_Cleaned.csv')

# Drop missing SMILES
data = data.dropna(subset=['SMILES_Standardized'])

smiles_list = data['SMILES_Standardized'].astype(str).tolist()

bit_info_all = []

for smi in smiles_list:
    if not isinstance(smi, str) or smi.strip() == "":
        continue
    
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    
    bit_info = {}
    
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048,
        bitInfo=bit_info
    )
    
    bit_info_all.append((mol, bit_info))

print(len(bit_info_all))
print(bit_info_all[0])

from rdkit.Chem import Draw

def draw_bit(mol, atom_idx, radius):
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    atoms = set()
    
    for bond_idx in env:
        bond = mol.GetBondWithIdx(bond_idx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    
    return Draw.MolToImage(mol, highlightAtoms=list(atoms))

mol, bit_info = bit_info_all[0]

# Example: bit 1380
atom_idx, radius = bit_info[1380][0]

img = draw_bit(mol, atom_idx, radius)
img.show()