"""
CLI for Mixture DCN Predictor

Interactive command-line interface for predicting Derived Cetane Number (DCN)
of fuel mixtures using SMILES input.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor


def validate_smiles(smiles: str) -> bool:
    """Validate that a SMILES string is parseable by RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def get_molecule_info(smiles: str) -> dict:
    """Get basic information about a molecule from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
        'mol_weight': Descriptors.MolWt(mol),
        'num_atoms': mol.GetNumAtoms()
    }


def get_user_config():
    """
    Collect user inputs for mixture DCN prediction.
    Returns mixture composition (SMILES + mole fractions).
    """
    print("\n" + "="*70)
    print("MIXTURE DCN PREDICTOR - Configuration")
    print("="*70)
    
    # Get number of components
    while True:
        number_of_components = input("\nEnter number of components in the mixture (e.g., 2-10): ").strip()
        
        if not number_of_components.isdigit():
            print("❌ Invalid input. Please enter a positive integer.")
            continue
        
        num_comp = int(number_of_components)
        
        if num_comp < 2:
            print("❌ Mixture must have at least 2 components.")
            print("   For single-component prediction, use the pure component predictor.")
            continue
        
        if num_comp > 11:
            print("⚠ Warning: Model trained on max 11 components. Limiting to 11.")
            num_comp = 11
        
        break
    
    print(f"\n✓ Configuring mixture with {num_comp} components")
    print("-" * 70)
    
    # Collect SMILES and mole fractions
    components = []
    total_fraction = 0.0
    
    for i in range(num_comp):
        print(f"\n📋 Component {i+1}/{num_comp}:")
        print("-" * 40)
        
        # Get SMILES
        while True:
            smiles = input(f"  Enter SMILES: ").strip()
            
            if not smiles:
                print("  ❌ SMILES cannot be empty.")
                continue
            
            if not validate_smiles(smiles):
                print(f"  ❌ Invalid SMILES: '{smiles}'")
                print("     Please enter a valid SMILES string.")
                continue
            
            # Show molecule info
            mol_info = get_molecule_info(smiles)
            print(f"  ✓ Valid molecule: {mol_info['formula']} (MW: {mol_info['mol_weight']:.2f})")
            break
        
        # Get mole fraction
        while True:
            # For last component, auto-calculate remaining fraction
            if i == num_comp - 1:
                remaining = 1.0 - total_fraction
                print(f"  Mole fraction: {remaining:.4f} (auto-calculated)")
                mole_frac = remaining
                break
            
            frac_input = input(f"  Enter mole fraction (0-1, remaining: {1.0 - total_fraction:.4f}): ").strip()
            
            try:
                mole_frac = float(frac_input)
            except ValueError:
                print("  ❌ Invalid input. Please enter a number between 0 and 1.")
                continue
            
            if mole_frac < 0 or mole_frac > 1:
                print("  ❌ Mole fraction must be between 0 and 1.")
                continue
            
            if total_fraction + mole_frac > 1.0:
                print(f"  ❌ Total would exceed 1.0 (current total: {total_fraction:.4f})")
                continue
            
            if mole_frac == 0:
                print("  ⚠ Warning: Zero mole fraction. Component will have no effect.")
            
            break
        
        components.append({
            'smiles': smiles,
            'mole_fraction': mole_frac,
            'formula': mol_info['formula']
        })
        
        total_fraction += mole_frac
        print(f"  ✓ Component added (Total fraction so far: {total_fraction:.4f})")
    
    # Validate total
    if abs(total_fraction - 1.0) > 1e-6:
        print(f"\n⚠ Warning: Mole fractions sum to {total_fraction:.6f}, normalizing to 1.0")
        for comp in components:
            comp['mole_fraction'] /= total_fraction
    
    return components

