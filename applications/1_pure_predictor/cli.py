# applications/1_pure_predictor/cli.py

from rdkit import Chem
from rdkit.Chem import rdinchi


def get_user_config():
    """
    Collect user inputs for pure-component property prediction.
    SMILES-only input.
    """

    mode = input("Select prediction mode (1: Single, 2: Batch): ").strip()
    while mode not in {"1", "2"}:
        print("Invalid selection. Please choose 1 or 2.")
        mode = input("Select prediction mode (1: Single, 2: Batch): ").strip()

    if mode == "1":
        smiles = input("Enter SMILES string: ").strip()
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError("Invalid SMILES string.")
    else:
        smiles = input("Enter path to SMILES file: ").strip()

    return {
        "mode": mode,
        "smiles": smiles
    }

