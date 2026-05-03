"""
Binary Blending Tool — sweep logic, results display, and file I/O.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class BlendSweepResult:
    """Per-point predictions from a binary blending sweep."""
    additive_smiles: str
    base_fuel_type: str
    base_fuel_smiles: List[str]
    base_fuel_mole_fractions: List[float]
    mole_fractions: List[float]     # additive fraction at each sweep point
    dcn: List[Optional[float]]      # mixture DCN at each sweep point
    ysi: List[Optional[float]]      # mixture YSI at each sweep point

    def to_dict(self) -> dict:
        """JSON-serialisable form, ready for a frontend slider."""
        return {
            "additive_smiles": self.additive_smiles,
            "base_fuel_type": self.base_fuel_type,
            "base_fuel_smiles": self.base_fuel_smiles,
            "base_fuel_mole_fractions": self.base_fuel_mole_fractions,
            "sweep": [
                {"mole_fraction": x, "dcn": dcn, "ysi": ysi}
                for x, dcn, ysi in zip(self.mole_fractions, self.dcn, self.ysi)
            ],
        }


def run_blend_sweep(
    additive_smiles: str,
    base_fuel_type: str = "fossil_diesel",
    base_fuel_smiles: Optional[List[str]] = None,
    base_fuel_mole_fractions: Optional[List[float]] = None,
    min_fraction: float = 0.0,
    max_fraction: float = 0.30,
    n_steps: int = 20,
    verbose: bool = True,
) -> BlendSweepResult:
    """
    Sweep additive mole fraction from min_fraction to max_fraction and predict
    mixture DCN and YSI at each composition.

    At each sweep point the additive takes fraction f and the base fuel
    components are scaled to fill the remaining (1 - f), preserving their
    relative proportions.

    Args:
        additive_smiles: SMILES of the additive to blend in.
        base_fuel_type: "fossil_diesel", "biodiesel", or "custom".
        base_fuel_smiles: Component SMILES — required when base_fuel_type="custom".
        base_fuel_mole_fractions: Mole fractions — required when base_fuel_type="custom".
        min_fraction: Minimum additive mole fraction (inclusive, ≥ 0).
        max_fraction: Maximum additive mole fraction (inclusive, ≤ 1).
        n_steps: Number of evenly-spaced sweep points (≥ 2).
        verbose: Print per-point progress to stdout.

    Returns:
        BlendSweepResult with per-point DCN and YSI predictions.
    """
    from rdkit import Chem
    from core.base_fuel_library import BaseFuelLibrary
    from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
    from core.predictors.pure_component.property_predictor import PropertyPredictor
    from core.blending.blending_law import blend_ysi_mass_weighted

    if Chem.MolFromSmiles(additive_smiles) is None:
        raise ValueError(f"Invalid additive SMILES: {additive_smiles}")
    if not 0.0 <= min_fraction < max_fraction <= 1.0:
        raise ValueError("Require 0 ≤ min_fraction < max_fraction ≤ 1.")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2.")

    # Resolve base fuel composition
    if base_fuel_type == "custom":
        if base_fuel_smiles is None or base_fuel_mole_fractions is None:
            raise ValueError("Custom base fuel requires smiles and fractions.")
        base_smiles = base_fuel_smiles
        base_fracs = base_fuel_mole_fractions
    else:
        base_smiles, base_fracs = BaseFuelLibrary.get_base_fuel(base_fuel_type)

    if 1 + len(base_smiles) > 11:
        raise ValueError(
            f"Total components ({1 + len(base_smiles)}) exceed the GNN limit of 11."
        )

    sweep_fractions = list(np.linspace(min_fraction, max_fraction, n_steps))

    if verbose:
        print("\nInitialising predictors...")

    dcn_predictor = MixtureDCNPredictor()
    prop_predictor = PropertyPredictor()

    # Pre-compute pure-component YSI for additive and all base components (one batch call)
    all_smiles = [additive_smiles] + base_smiles
    props = prop_predictor.predict_all_properties(all_smiles)
    pure_ysi: List[Optional[float]] = props.get("ysi", [None] * len(all_smiles))
    additive_ysi: Optional[float] = pure_ysi[0]
    base_ysi: List[Optional[float]] = pure_ysi[1:]

    if verbose:
        print(f"\nBinary Blending Sweep")
        print(f"  Additive : {additive_smiles}")
        print(f"  Base fuel: {base_fuel_type} ({len(base_smiles)} components)")
        print(f"  Sweep    : {min_fraction:.3f} → {max_fraction:.3f}  ({n_steps} points)")
        add_ysi_str = f"{additive_ysi:.2f}" if additive_ysi is not None else "N/A"
        print(f"  Additive pure YSI: {add_ysi_str}\n")

    dcn_results: List[Optional[float]] = []
    ysi_results: List[Optional[float]] = []

    for idx, f in enumerate(sweep_fractions):
        adj_base_fracs = [bf * (1.0 - f) for bf in base_fracs]
        mixture_smiles = [additive_smiles] + base_smiles
        mixture_fracs = [f] + adj_base_fracs

        # DCN: always include additive so mixture size stays constant across the sweep
        try:
            dcn = dcn_predictor.predict_mixture_dcn(mixture_smiles, mixture_fracs)
        except Exception:
            dcn = None

        # YSI: exclude additive at f=0 so a failed additive YSI doesn't poison the baseline
        if f == 0.0:
            ysi = blend_ysi_mass_weighted(base_smiles, base_fracs, base_ysi)
        else:
            ysi = blend_ysi_mass_weighted(
                mixture_smiles, mixture_fracs, [additive_ysi] + base_ysi
            )

        dcn_results.append(dcn)
        ysi_results.append(ysi)

        if verbose:
            dcn_str = f"{dcn:.2f}" if dcn is not None else "N/A "
            ysi_str = f"{ysi:.2f}" if ysi is not None else "N/A "
            print(f"  [{idx+1:>2}/{n_steps}]  x_add={f:.4f}  DCN={dcn_str}  YSI={ysi_str}")

    if verbose:
        print(f"\n✓ Sweep complete.")

    return BlendSweepResult(
        additive_smiles=additive_smiles,
        base_fuel_type=base_fuel_type,
        base_fuel_smiles=base_smiles,
        base_fuel_mole_fractions=base_fracs,
        mole_fractions=sweep_fractions,
        dcn=dcn_results,
        ysi=ysi_results,
    )


def display_sweep_results(result: BlendSweepResult) -> None:
    """Print a formatted table of sweep results."""
    print("\n" + "=" * 70)
    print("BLEND SWEEP RESULTS")
    print("=" * 70)
    print(f"  Additive : {result.additive_smiles}")
    print(f"  Base fuel: {result.base_fuel_type}")
    print()
    print(f"  {'x_additive':>12}  {'DCN':>10}  {'YSI':>10}")
    print("  " + "-" * 38)
    for x, dcn, ysi in zip(result.mole_fractions, result.dcn, result.ysi):
        dcn_str = f"{dcn:10.2f}" if dcn is not None else "       N/A"
        ysi_str = f"{ysi:10.2f}" if ysi is not None else "       N/A"
        print(f"  {x:12.4f}  {dcn_str}  {ysi_str}")
    print("=" * 70)


def save_results(result: BlendSweepResult, filename: Optional[str] = None) -> None:
    """Save sweep results to CSV and JSON files."""
    import csv
    import json

    if filename is None:
        safe = result.additive_smiles[:20].replace("/", "_").replace("\\", "_")
        filename = f"blend_sweep_{safe}"

    csv_path = filename + ".csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mole_fraction", "dcn", "ysi"])
        writer.writeheader()
        for x, dcn, ysi in zip(result.mole_fractions, result.dcn, result.ysi):
            writer.writerow({"mole_fraction": x, "dcn": dcn, "ysi": ysi})
    print(f"✓ CSV  saved → '{csv_path}'")

    json_path = filename + ".json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"✓ JSON saved → '{json_path}'")
