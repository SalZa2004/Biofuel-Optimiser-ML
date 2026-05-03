"""
Binary Blending Tool CLI

Collects additive SMILES, base fuel selection, and sweep parameters
from the user interactively.
"""

from typing import Optional, List, Tuple


def _get_additive_smiles() -> str:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    print("\nAdditive Configuration")
    print("-" * 40)

    while True:
        smiles = input("Enter additive SMILES: ").strip()
        if not smiles:
            print("❌ SMILES cannot be empty.")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"❌ Invalid SMILES: '{smiles}'. Please try again.")
            continue

        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        mw = Descriptors.MolWt(mol)
        print(f"  ✓ Valid molecule: {formula} (MW: {mw:.2f} g/mol)")
        return smiles


def _get_base_fuel() -> Tuple[str, Optional[List[str]], Optional[List[float]]]:
    from applications.mixture_aware_generator.cli import get_custom_base_fuel

    print("\nBase Fuel Selection:")
    print("1. Fossil diesel (typical petroleum diesel)")
    print("2. Biodiesel (FAME)")
    print("3. Custom (enter your own composition)")

    while True:
        choice = input("\nSelect base fuel (1-3): ").strip()

        if choice == "1":
            print("✓ Using fossil diesel")
            return "fossil_diesel", None, None

        elif choice == "2":
            print("✓ Using biodiesel")
            return "biodiesel", None, None

        elif choice == "3":
            smiles, fracs = get_custom_base_fuel()
            if smiles is None:
                print("  → Defaulting to fossil diesel.")
                return "fossil_diesel", None, None
            return "custom", smiles, fracs

        else:
            print("❌ Invalid choice. Enter 1, 2, or 3.")


def _get_sweep_config() -> Tuple[float, float, int]:
    print("\nSweep Configuration")
    print("-" * 40)
    print("Define the range of additive mole fractions to evaluate.")

    while True:
        try:
            raw = input("Min additive mole fraction [default 0.00]: ").strip()
            min_frac = float(raw) if raw else 0.0
            if not 0.0 <= min_frac < 1.0:
                print("❌ Min fraction must be in [0, 1).")
                continue
            break
        except ValueError:
            print("❌ Please enter a number.")

    while True:
        try:
            raw = input("Max additive mole fraction [default 0.30]: ").strip()
            max_frac = float(raw) if raw else 0.30
            if not min_frac < max_frac <= 1.0:
                print(f"❌ Max fraction must be greater than {min_frac:.3f} and ≤ 1.")
                continue
            break
        except ValueError:
            print("❌ Please enter a number.")

    while True:
        try:
            raw = input("Number of sweep steps [default 20]: ").strip()
            n_steps = int(raw) if raw else 20
            if n_steps < 2:
                print("❌ Need at least 2 steps.")
                continue
            break
        except ValueError:
            print("❌ Please enter a whole number.")

    return min_frac, max_frac, n_steps


def get_user_config() -> dict:
    """Interactive CLI — returns a config dict that maps directly to run_blend_sweep kwargs."""
    print("=" * 70)
    print("BINARY BLENDING TOOL")
    print("=" * 70)
    print("\nPredict how DCN and YSI change as a single additive is blended")
    print("into a base fuel across a range of mole fractions.")

    additive_smiles = _get_additive_smiles()
    base_fuel_type, base_fuel_smiles, base_fuel_mole_fractions = _get_base_fuel()
    min_frac, max_frac, n_steps = _get_sweep_config()

    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"  Additive SMILES : {additive_smiles}")
    print(f"  Base fuel       : {base_fuel_type}")
    if base_fuel_smiles:
        print(f"  Base components : {len(base_fuel_smiles)}")
    print(f"  Sweep           : {min_frac:.3f} → {max_frac:.3f}  ({n_steps} steps)")
    print("=" * 70)

    return {
        "additive_smiles": additive_smiles,
        "base_fuel_type": base_fuel_type,
        "base_fuel_smiles": base_fuel_smiles,
        "base_fuel_mole_fractions": base_fuel_mole_fractions,
        "min_fraction": min_frac,
        "max_fraction": max_frac,
        "n_steps": n_steps,
    }
