from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from PIL import Image
import io

mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC")

bit_info = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=bit_info)

bit_id = 1143

if bit_id in bit_info:
    img = Draw.DrawMorganBit(mol, bit_id, bit_info)

    # Case 1: SVG string
    if isinstance(img, str):
        with open("bit_1143_structure.svg", "w") as f:
            f.write(img)
        print("Saved as SVG (bit_1143_structure.svg)")

    # Case 2: bytes (PNG stream)
    elif isinstance(img, bytes):
        img = Image.open(io.BytesIO(img))
        img.save("bit_1143_structure.png")
        print("Saved as PNG")

    # Case 3: PIL image
    else:
        img.save("bit_1143_structure.png")
        print("Saved as PNG")

else:
    print(f"Bit {bit_id} was not found in this molecule.")