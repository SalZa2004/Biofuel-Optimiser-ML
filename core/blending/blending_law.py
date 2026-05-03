from typing import List, Dict, Optional
import math


def blend_bp_riazi_daubert(
    smiles: List[str],
    mole_fracs: List[float],
    densities: List[Optional[float]],
) -> Optional[float]:
    """
    Riazi-Daubert mixture boiling point (°C) from mole fractions and densities.

    Computes mixture average molecular weight M and specific gravity S, then
    applies the Riazi-Daubert correlation:
      M 70–300:  Tb = 3.76587 * exp(3.7741e-3*M + 2.98404*S - 4.25288e-3*M*S) * M^0.40167 * S^-1.58262
      M 300–700: Tb = 9.3369  * exp(1.6514e-4*M + 1.4103*S  - 7.5152e-4*M*S)  * M^0.5369  * S^-0.7276

    Tb is in Kelvin; returned value is in °C.
    S is specific gravity at 60°F (15.6°C) ≈ density in g/cm³.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    if any(d is None for d in densities):
        return None

    mol_weights = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol_weights.append(Descriptors.ExactMolWt(mol))

    # Mole-fraction weighted average molecular weight
    M_mix = sum(x * mw for x, mw in zip(mole_fracs, mol_weights))

    # Mass fractions
    masses = [x * mw for x, mw in zip(mole_fracs, mol_weights)]
    total_mass = sum(masses)
    if total_mass == 0:
        return None
    w = [m / total_mass for m in masses]

    # Mixture density via volume-additive rule: rho_mix = 1 / sum(w_i / rho_i)
    S_mix = 1.0 / sum(wi / rho for wi, rho in zip(w, densities))

    try:
        if M_mix <= 300:
            Tb_K = (3.76587
                    * math.exp(3.7741e-3 * M_mix + 2.98404 * S_mix - 4.25288e-3 * M_mix * S_mix)
                    * M_mix ** 0.40167
                    * S_mix ** (-1.58262))
        else:
            Tb_K = (9.3369
                    * math.exp(1.6514e-4 * M_mix + 1.4103 * S_mix - 7.5152e-4 * M_mix * S_mix)
                    * M_mix ** 0.5369
                    * S_mix ** (-0.7276))
    except (ValueError, OverflowError):
        return None

    return Tb_K - 273.15


def blend_ysi_mass_weighted(
    smiles: List[str],
    mole_fracs: List[float],
    ysi_values: List[Optional[float]],
) -> Optional[float]:
    """
    Mass-fraction-weighted mixture YSI from mole fractions.

    Converts mole fractions to mass fractions using RDKit exact molecular
    weights, then applies the linear YSI blending law.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    if any(v is None for v in ysi_values):
        return None

    mol_weights = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol_weights.append(Descriptors.ExactMolWt(mol))

    masses = [x * mw for x, mw in zip(mole_fracs, mol_weights)]
    total_mass = sum(masses)
    if total_mass == 0:
        return None
    mass_fracs = [m / total_mass for m in masses]

    return sum(y * mf for y, mf in zip(ysi_values, mass_fracs))


# Carbon-type YSI contributions C_j (Table 5)
YSI_CARBON_CONTRIBUTIONS: Dict[str, float] = {
    "ct1":  -4.49,   # CH3
    "ct2":   1.52,   # n-alkane CH2
    "ct3":   2.11,   # iso-alkane CH
    "ct4":   2.71,   # cyclo-alkane CH2
    "ct5":  19.02,   # cyclo-alkane → alkyl-chain CH
    "ct6":  13.07,   # cyclo-alkane → cyclo-alkane CH  (ring junction)
    "ct7":  20.61,   # aromatic CH
    "ct8":  70.23,   # aromatic → alkyl-chain C
    "ct9":  85.55,   # aromatic → cyclo-alkane C  (fused)
    "ct10": 112.09,  # aromatic → aromatic C  (fused)
    "ct11":  23.70,  # aliphatic quaternary C  (0 H, non-aromatic)
}


def count_carbon_types(smiles: str) -> Optional[Dict[str, int]]:
    """
    Classify every carbon atom in a molecule into one of 11 carbon types.

    Classification rules
    --------------------
    Aromatic carbons (in aromatic ring):
      ct7  : 1 H
      ct10 : 0 H, in ≥2 aromatic rings  (aromatic–aromatic junction, e.g. naphthalene)
      ct9  : 0 H, in ≥1 non-aromatic ring as well  (fused to cycloalkane, e.g. tetralin)
      ct8  : 0 H, otherwise  (single aromatic ring with alkyl substituent, e.g. toluene)

    Non-aromatic ring carbons:
      ct4  : 2 H  (ring CH2)
      ct5  : 1 H, at least one carbon neighbour not in any ring  (ring–chain junction)
      ct6  : 1 H, all carbon neighbours in rings  (ring–ring junction, e.g. decalin)
      ct11 : 0 H  (quaternary ring carbon)

    Acyclic non-aromatic carbons:
      ct1  : 3 H  (methyl)
      ct2  : 2 H  (linear CH2)
      ct3  : 1 H  (branched CH)
      ct11 : 0 H  (quaternary carbon)
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ring_info = mol.GetRingInfo()
    counts: Dict[str, int] = {ct: 0 for ct in YSI_CARBON_CONTRIBUTIONS}

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue

        n_H       = atom.GetTotalNumHs()
        aromatic  = atom.GetIsAromatic()
        in_ring   = atom.IsInRing()
        idx       = atom.GetIdx()

        atom_rings = [r for r in ring_info.AtomRings() if idx in r]
        n_arom_rings = sum(
            1 for r in atom_rings
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
        )
        n_nonarom_rings = len(atom_rings) - n_arom_rings

        if aromatic:
            if n_H == 1:
                counts["ct7"] += 1
            else:  # n_H == 0
                if n_arom_rings >= 2:
                    counts["ct10"] += 1
                elif n_nonarom_rings >= 1:
                    counts["ct9"] += 1
                else:
                    counts["ct8"] += 1

        elif in_ring:
            if n_H == 2:
                counts["ct4"] += 1
            elif n_H == 1:
                has_chain_nb = any(
                    not nb.IsInRing() and nb.GetAtomicNum() == 6
                    for nb in atom.GetNeighbors()
                )
                counts["ct5" if has_chain_nb else "ct6"] += 1
            else:
                counts["ct11"] += 1

        else:  # acyclic non-aromatic
            if   n_H == 3: counts["ct1"]  += 1
            elif n_H == 2: counts["ct2"]  += 1
            elif n_H == 1: counts["ct3"]  += 1
            else:          counts["ct11"] += 1

    return counts


def ysi_from_carbon_types(smiles: str) -> Optional[float]:
    """
    Predict pure-component YSI via carbon-type contributions (Eq 7b).

    YSI_PC = Σ_j  N_j × C_j
    """
    counts = count_carbon_types(smiles)
    if counts is None:
        return None
    return sum(n * YSI_CARBON_CONTRIBUTIONS[ct] for ct, n in counts.items())


def rescale_das_to_unified(ysi_das: float) -> float:
    """Convert YSI from Das et al. 2017 scale to unified YSI scale.
    Fit from 23 overlapping compounds, R²=0.95
    """
    return 1.1743 * ysi_das + 34.9322


def blend_ysi_carbon_type(
    smiles: List[str],
    mass_fracs: List[float],
) -> Optional[float]:
    """
    Predict mixture YSI using the carbon-type contribution method (Eq 7a + 7b),
    rescaled from Das et al. 2017 scale to unified YSI scale.

    YSI_SM = rescale( Σ_i  W_i × YSI_PC_i )

    where W_i are mass fractions and YSI_PC_i is computed from carbon types.
    """
    ysi_values = [ysi_from_carbon_types(smi) for smi in smiles]
    if any(v is None for v in ysi_values):
        return None
    blended = sum(w * y for w, y in zip(mass_fracs, ysi_values))
    return rescale_das_to_unified(blended)
