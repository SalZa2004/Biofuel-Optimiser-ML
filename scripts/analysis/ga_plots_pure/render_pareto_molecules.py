import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw
from pathlib import Path
import re

OUTPUT_DIR = Path(__file__).parent / "pareto_molecules"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = Path(__file__).parent / "ga_pareto_front.csv"


def get_iupac_name(smiles: str) -> str:
    results = pcp.get_compounds(smiles, namespace="smiles")
    if results:
        name = results[0].iupac_name
        if name:
            return name
    return None


def safe_filename(name: str) -> str:
    # Replace characters that are invalid in filenames
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def main():
    df = pd.read_csv(CSV_PATH)

    for _, row in df.iterrows():
        rank = int(row["rank"])
        smiles = row["smiles"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  Rank {rank}: invalid SMILES, skipping")
            continue

        print(f"Rank {rank}: fetching IUPAC name for {smiles} ...")
        iupac = get_iupac_name(smiles)

        if iupac is None:
            print(f"  -> no IUPAC name found, using SMILES as fallback")
            label = safe_filename(smiles)
        else:
            print(f"  -> {iupac}")
            label = safe_filename(iupac)

        filename = OUTPUT_DIR / f"{rank}_{label}.png"
        img = Draw.MolToImage(mol, size=(400, 300))
        img.save(filename)
        print(f"  Saved: {filename.name}")

    print(f"\nDone. {len(df)} images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
