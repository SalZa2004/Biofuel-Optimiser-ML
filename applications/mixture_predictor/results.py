from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor


def display_mixture_summary(components):
    """Display a formatted summary of the mixture composition."""
    print("\n" + "="*70)
    print("MIXTURE COMPOSITION SUMMARY")
    print("="*70)

    print(f"\n{'Component':<12} {'Formula':<15} {'SMILES':<30} {'Mole Frac':<10}")
    print("-" * 70)

    for i, comp in enumerate(components, 1):
        print(f"{i:<12} {comp['formula']:<15} {comp['smiles'][:28]:<30} {comp['mole_fraction']:.4f}")

    total = sum(c['mole_fraction'] for c in components)
    print("-" * 70)
    print(f"{'TOTAL':<12} {'':<15} {'':<30} {total:.4f}")
    print("="*70)


def predict_mixture_properties(components):
    """Predict DCN, YSI, boiling point, and density for the mixture.

    Returns a dict with keys 'dcn', 'ysi', 'bp', 'density' (all floats or None).
    """
    from core.predictors.pure_component.property_predictor import PropertyPredictor
    from core.blending.blending_law import (
        blend_ysi_mass_weighted,
        blend_bp_riazi_daubert,
        blend_density,
    )

    smiles_list = [c['smiles'] for c in components]
    mole_fractions = [c['mole_fraction'] for c in components]

    print("\nPredicting mixture properties...")
    print("-" * 70)

    try:
        # --- DCN (GNN ensemble) ---
        dcn_predictor = MixtureDCNPredictor()
        dcn = dcn_predictor.predict_mixture_dcn(smiles_list, mole_fractions)

        # --- Pure-component YSI and density (needed for blending laws) ---
        prop_predictor = PropertyPredictor()
        props = prop_predictor.predict_all_properties(smiles_list)

        ysi_values = props.get('ysi', [None] * len(smiles_list))
        density_values = props.get('density', [None] * len(smiles_list))  # kg/m³

        # --- Mixture YSI (mass-fraction weighted blending law) ---
        mixture_ysi = blend_ysi_mass_weighted(smiles_list, mole_fractions, ysi_values)

        # --- Mixture BP and density (Riazi-Daubert / volume-additive, need g/cm³) ---
        densities_gcc = [d / 1000.0 if d is not None else None for d in density_values]
        mixture_bp = blend_bp_riazi_daubert(smiles_list, mole_fractions, densities_gcc)
        density_gcc = blend_density(smiles_list, mole_fractions, densities_gcc)
        mixture_density = density_gcc * 1000.0 if density_gcc is not None else None  # → kg/m³

        predictions = {
            'dcn': dcn,
            'ysi': mixture_ysi,
            'bp': mixture_bp,
            'density': mixture_density,
        }

        print(f"\n✅ PREDICTION SUCCESSFUL!")
        print("=" * 70)
        print(f"  Derived Cetane Number (DCN):  {dcn:.2f}" if dcn is not None else "  DCN:     N/A")
        print(f"  Mixture YSI:                  {mixture_ysi:.2f}" if mixture_ysi is not None else "  YSI:     N/A")
        print(f"  Mixture Boiling Point (°C):   {mixture_bp:.1f}" if mixture_bp is not None else "  BP:      N/A")
        print(f"  Mixture Density (kg/m³):      {mixture_density:.1f}" if mixture_density is not None else "  Density: N/A")
        print("=" * 70)

        return predictions

    except Exception as e:
        print(f"\n❌ PREDICTION FAILED!")
        print(f"Error: {str(e)}")
        print("\nPlease check your inputs and try again.")
        return None


def save_results(components, predictions):
    """Optionally save results to a file."""
    save = input("\nWould you like to save these results to a file? (y/n): ").strip().lower()

    if save != 'y':
        return

    filename = input("Enter filename (default: mixture_prediction.txt): ").strip()
    if not filename:
        filename = "mixture_prediction.txt"

    if not filename.endswith('.txt'):
        filename += '.txt'

    try:
        with open(filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MIXTURE PROPERTY PREDICTION RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write("Mixture Composition:\n")
            f.write("-" * 70 + "\n")
            for i, comp in enumerate(components, 1):
                f.write(f"Component {i}:\n")
                f.write(f"  SMILES:        {comp['smiles']}\n")
                f.write(f"  Formula:       {comp['formula']}\n")
                f.write(f"  Mole Fraction: {comp['mole_fraction']:.4f}\n\n")

            f.write("-" * 70 + "\n\n")
            f.write("Predicted Properties:\n")

            dcn = predictions.get('dcn')
            ysi = predictions.get('ysi')
            bp = predictions.get('bp')
            density = predictions.get('density')

            f.write(f"  Derived Cetane Number (DCN):  {dcn:.2f}\n" if dcn is not None else "  DCN:                          N/A\n")
            f.write(f"  Mixture YSI:                  {ysi:.2f}\n" if ysi is not None else "  Mixture YSI:                  N/A\n")
            f.write(f"  Mixture Boiling Point (°C):   {bp:.1f}\n" if bp is not None else "  Mixture Boiling Point:        N/A\n")
            f.write(f"  Mixture Density (kg/m³):      {density:.1f}\n" if density is not None else "  Mixture Density:              N/A\n")
            f.write("=" * 70 + "\n")

        print(f"✓ Results saved to '{filename}'")

    except Exception as e:
        print(f"Failed to save file: {e}")
