"""CLI for the biofuel screening tool."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


@dataclass
class ScreeningConfig:
    csv_path: str
    mode: str  # "pure_component" or "mixture"
    target_cn: float
    filters: Dict = field(default_factory=lambda: {
        "bp": (60.0, 250.0),
        "density": (720.0, None),
        "lhv": (30.0, None),
        "dynamic_viscosity": (2.0, None),
    })


def _get_csv_path() -> str:
    import os
    while True:
        path = input("Path to CSV file (must have a 'smiles' column): ").strip()
        if not path:
            print("  Path cannot be empty.")
            continue
        if not os.path.isfile(path):
            print(f"  File not found: {path}")
            continue
        return path


def _get_filter_bounds(label: str, default_lo, default_hi) -> Tuple:
    lo_str = input(f"    Min {label} [{default_lo}]: ").strip()
    hi_str = input(f"    Max {label} [{default_hi if default_hi is not None else 'none'}]: ").strip()
    try:
        lo = float(lo_str) if lo_str else default_lo
    except ValueError:
        lo = default_lo
    try:
        hi = float(hi_str) if hi_str else default_hi
    except ValueError:
        hi = default_hi
    return (lo, hi)


def _get_filter_config() -> Dict:
    print("\n  Boiling Point (°C):")
    bp = _get_filter_bounds("BP (°C)", 60.0, 250.0)
    print("  Density (kg/m³):")
    density_lo, _ = _get_filter_bounds("density", 720.0, None)
    print("  LHV (MJ/kg):")
    lhv_lo, _ = _get_filter_bounds("LHV", 30.0, None)
    print("  Dynamic Viscosity (cSt):")
    visc_lo, _ = _get_filter_bounds("viscosity", 2.0, None)
    return {
        "bp": bp,
        "density": (density_lo, None),
        "lhv": (lhv_lo, None),
        "dynamic_viscosity": (visc_lo, None),
    }


def _get_custom_base_fuel():
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    print("\n" + "=" * 70)
    print("CUSTOM BASE FUEL CONFIGURATION")
    print("=" * 70)

    while True:
        s = input("\nNumber of base fuel components (2-10): ").strip()
        if not s.isdigit():
            print("  Enter a positive integer.")
            continue
        n = int(s)
        if n < 2:
            print("  Must have at least 2 components.")
            continue
        if n > 10:
            if input("  Warning: many components may slow predictions. Continue? (y/n): ").strip().lower() != "y":
                continue
        break

    smiles_list, frac_list, total = [], [], 0.0

    for i in range(n):
        print(f"\nComponent {i + 1}/{n}:")

        while True:
            smi = input("  SMILES: ").strip()
            if not smi:
                print("  Cannot be empty.")
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"  Invalid SMILES.")
                continue
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            mw = Descriptors.MolWt(mol)
            print(f"  {formula}  (MW: {mw:.2f} g/mol)")
            break

        while True:
            if i == n - 1:
                rem = 1.0 - total
                print(f"  Remaining fraction: {rem:.4f}")
                if input("  Use remaining? (y/n): ").strip().lower() == "y":
                    frac = rem
                    break
            try:
                frac = float(input(f"  Mole fraction (remaining: {1.0 - total:.4f}): ").strip())
            except ValueError:
                print("  Invalid number.")
                continue
            if frac <= 0 or total + frac > 1.0001:
                print("  Invalid fraction.")
                continue
            break

        smiles_list.append(smi)
        frac_list.append(frac)
        total += frac

    if abs(total - 1.0) > 1e-6:
        frac_list = [f / total for f in frac_list]

    print("\n" + "=" * 70)
    print("CUSTOM BASE FUEL SUMMARY")
    print("=" * 70)
    for j, (s, f) in enumerate(zip(smiles_list, frac_list), 1):
        mol = Chem.MolFromSmiles(s)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        print(f"  {j}. {formula:<15} {s[:40]:<42} {f:.4f}")
    print(f"  {'TOTAL':<58} {sum(frac_list):.4f}")
    print("=" * 70)

    if input("\nUse this custom base fuel? (y/n): ").strip().lower() != "y":
        print("  Cancelled — defaulting to fossil diesel.")
        return None, None
    return smiles_list, frac_list


def get_user_config() -> ScreeningConfig:
    print("=" * 70)
    print("BIOFUEL SCREENING TOOL")
    print("=" * 70)

    csv_path = _get_csv_path()

    print("\nScreening Mode:")
    print("  1. Pure component screening")
    print("  2. Mixture / blend screening")
    while True:
        choice = input("Select mode (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  Please enter 1 or 2.")
    mode = "pure_component" if choice == "1" else "mixture"
    print(f"  Mode: {'Pure Component' if mode == 'pure_component' else 'Mixture'} Screening")

    while True:
        try:
            target_cn = float(input("\nTarget cetane number (CN): ").strip())
            if target_cn <= 0:
                print("  CN must be positive.")
                continue
            break
        except ValueError:
            print("  Invalid input.")

    if mode == "pure_component":
        print("\nProperty Constraint Filters:")
        print("  Defaults: BP 60–250 °C | density ≥ 720 kg/m³ | LHV ≥ 30 MJ/kg | viscosity ≥ 2 cSt")
        if input("Use defaults? (y/n) [y]: ").strip().lower() == "n":
            filters = _get_filter_config()
        else:
            filters = {
                "bp": (60.0, 250.0),
                "density": (720.0, None),
                "lhv": (30.0, None),
                "dynamic_viscosity": (2.0, None),
            }

        print("\n" + "=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"  Mode:      Pure Component Screening")
        print(f"  CSV:       {csv_path}")
        print(f"  Target CN: {target_cn}")
        for prop, (lo, hi) in filters.items():
            bounds = f"{lo if lo is not None else '—'} – {hi if hi is not None else '—'}"
            print(f"  {prop}: {bounds}")
        print("=" * 70)

        return ScreeningConfig(csv_path=csv_path, mode=mode, target_cn=target_cn, filters=filters)

    # --- mixture mode ---
    # CSV must follow mixture_database.csv format:
    #   Name, fuel_1_smiles, fraction_fuel_1, fuel_2_smiles, fraction_fuel_2, ...
    print("\n  CSV must follow the mixture_database.csv format")
    print("  (columns: Name, fuel_1_smiles, fraction_fuel_1, fuel_2_smiles, ...)")

    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"  Mode:      Mixture Screening")
    print(f"  CSV:       {csv_path}")
    print(f"  Target CN: {target_cn}")
    print("=" * 70)

    return ScreeningConfig(csv_path=csv_path, mode=mode, target_cn=target_cn)
