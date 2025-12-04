from crem.crem import mutate_mol, grow_mol, link_mols
from rdkit import Chem


m = Chem.MolFromSmiles('c1cc(OC)ccc1C')  # methoxytoluene
mols = list(mutate_mol(m, db_name='replacements.db', max_size=1))
