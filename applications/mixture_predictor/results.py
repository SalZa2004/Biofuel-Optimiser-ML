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


def predict_dcn(components):
    """Make DCN prediction for the mixture."""
    print("\nPredicting Derived Cetane Number (DCN)...")
    print("-" * 70)
    
    try:
        # Initialize predictor
        predictor = MixtureDCNPredictor()
        
        # Extract SMILES and fractions
        smiles_list = [c['smiles'] for c in components]
        mole_fractions = [c['mole_fraction'] for c in components]
        
        # Predict
        dcn = predictor.predict_mixture_dcn(smiles_list, mole_fractions)
        
        print(f"\n✅ PREDICTION SUCCESSFUL!")
        print("="*70)
        print(f"  Derived Cetane Number (DCN): {dcn:.2f}")
        print("="*70)
    


        return dcn
    
    except Exception as e:
        print(f"\n❌ PREDICTION FAILED!")
        print(f"Error: {str(e)}")
        print("\nPlease check your inputs and try again.")
        return None


def save_results(components, dcn):
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
            f.write("="*70 + "\n")
            f.write("MIXTURE DCN PREDICTION RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write("Mixture Composition:\n")
            f.write("-"*70 + "\n")
            for i, comp in enumerate(components, 1):
                f.write(f"Component {i}:\n")
                f.write(f"  SMILES:        {comp['smiles']}\n")
                f.write(f"  Formula:       {comp['formula']}\n")
                f.write(f"  Mole Fraction: {comp['mole_fraction']:.4f}\n\n")
            
            f.write("-"*70 + "\n\n")
            f.write(f"Predicted DCN: {dcn:.2f}\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Results saved to '{filename}'")
    
    except Exception as e:
        print(f"Failed to save file: {e}")

