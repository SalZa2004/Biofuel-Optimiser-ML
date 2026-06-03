import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "scripts/analysis/ga_plots_pure/ga_pareto_front.csv"
OUTPUT_PATH = "scripts/analysis/ga_plots_pure/pareto_molecule_table.png"


def get_iupac_name(smiles):
    try:
        compounds = pcp.get_compounds(smiles, "smiles")
        if compounds and compounds[0].iupac_name:
            return compounds[0].iupac_name
        # fallback: try by inchi
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            compounds = pcp.get_compounds(smiles, "smiles", searchtype="similarity", listkey_count=1)
            if compounds and compounds[0].iupac_name:
                return compounds[0].iupac_name
    except Exception:
        pass
    return "N/A"


def smiles_to_img_array(smiles, size=(280, 200)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.ones((*size[::-1], 3), dtype=np.uint8) * 255
    img = Draw.MolToImage(mol, size=size)
    return np.array(img)


def main():
    df = pd.read_csv(DATA_PATH)
    smiles_list = df["smiles"].tolist()
    cn_list = df["cn"].tolist()
    ysi_list = df["ysi"].tolist()

    print(f"Fetching IUPAC names for {len(smiles_list)} molecules...")
    iupac_names = []
    for i, smi in enumerate(smiles_list):
        name = get_iupac_name(smi)
        iupac_names.append(name)
        print(f"  [{i+1}/{len(smiles_list)}] {smi[:30]:<30} -> {name}")

    n = len(smiles_list)
    fig_width = 22
    row_height = 3.2
    header_height = 0.6
    fig_height = header_height + n * row_height

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    col_widths = [0.18, 0.25, 0.28, 0.145, 0.145]
    col_labels = ["Structure", "IUPAC Name", "SMILES", "CN", "YSI"]
    col_colors = ["#2c3e50"] * 5
    header_y = 1.0 - header_height / fig_height

    # draw header
    x_pos = 0.0
    for lbl, w, color in zip(col_labels, col_widths, col_colors):
        ax_h = fig.add_axes([x_pos, header_y, w, header_height / fig_height])
        ax_h.set_facecolor(color)
        ax_h.text(
            0.5, 0.5, lbl,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="white",
            transform=ax_h.transAxes,
        )
        ax_h.set_xticks([])
        ax_h.set_yticks([])
        for spine in ax_h.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(1.5)
        x_pos += w

    # row background alternation
    row_bg = ["#f8f9fa", "#edf2f7"]

    for row_idx, (smi, iupac, cn, ysi) in enumerate(
        zip(smiles_list, iupac_names, cn_list, ysi_list)
    ):
        row_frac = row_height / fig_height
        row_bottom = header_y - (row_idx + 1) * row_frac
        bg = row_bg[row_idx % 2]

        x_pos = 0.0
        for col_idx, (w, lbl) in enumerate(zip(col_widths, col_labels)):
            ax = fig.add_axes([x_pos, row_bottom, w, row_frac])
            ax.set_facecolor(bg)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#dee2e6")
                spine.set_linewidth(0.8)

            if lbl == "Structure":
                img_arr = smiles_to_img_array(smi, size=(260, 190))
                ax.imshow(img_arr)
                ax.set_aspect("auto")
            elif lbl == "IUPAC Name":
                # wrap long names
                words = iupac.split("-")
                lines, cur = [], ""
                for w_part in words:
                    candidate = cur + ("-" if cur else "") + w_part
                    if len(candidate) > 22 and cur:
                        lines.append(cur + "-")
                        cur = w_part
                    else:
                        cur = candidate
                if cur:
                    lines.append(cur)
                wrapped = "\n".join(lines)
                ax.text(
                    0.5, 0.5, wrapped,
                    ha="center", va="center",
                    fontsize=9, color="#212529",
                    transform=ax.transAxes,
                    wrap=True,
                )
            elif lbl == "SMILES":
                # break SMILES at every 18 chars
                chunks = [smi[i:i+18] for i in range(0, len(smi), 18)]
                ax.text(
                    0.5, 0.5, "\n".join(chunks),
                    ha="center", va="center",
                    fontsize=8, color="#212529", family="monospace",
                    transform=ax.transAxes,
                )
            elif lbl == "CN":
                ax.text(
                    0.5, 0.5, f"{cn:.1f}",
                    ha="center", va="center",
                    fontsize=11, color="#212529", fontweight="bold",
                    transform=ax.transAxes,
                )
            elif lbl == "YSI":
                ax.text(
                    0.5, 0.5, f"{ysi:.1f}",
                    ha="center", va="center",
                    fontsize=11, color="#212529", fontweight="bold",
                    transform=ax.transAxes,
                )
            x_pos += col_widths[col_idx]

    fig.suptitle(
        "Pareto-Optimal Molecules: Structure, CN and YSI",
        fontsize=16, fontweight="bold", y=1.0 - 0.005,
        color="#2c3e50",
    )

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {OUTPUT_PATH}")

    # also print a clean text table
    print("\n" + "=" * 100)
    print(f"{'Rank':<5} {'CN':>7} {'YSI':>8}  {'IUPAC Name':<40} {'SMILES'}")
    print("=" * 100)
    for i, (smi, iupac, cn, ysi) in enumerate(
        zip(smiles_list, iupac_names, cn_list, ysi_list), 1
    ):
        print(f"{i:<5} {cn:>7.1f} {ysi:>8.2f}  {iupac:<40} {smi}")
    print("=" * 100)


if __name__ == "__main__":
    main()
