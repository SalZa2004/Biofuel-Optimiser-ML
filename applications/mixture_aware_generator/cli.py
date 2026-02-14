"""
Updated CLI with Mixture Optimization Support - INCLUDING CUSTOM BASE FUEL
"""
from core.config import EvolutionConfig, MixtureConfig


def get_user_config() -> EvolutionConfig:
    """Interactive CLI to get configuration from user."""
    
    print("="*70)
    print("MIXTURE-AWARE MOLECULE GENERATOR")
    print("="*70)
    
    return get_mixture_config()


def get_custom_base_fuel():
    """
    Interactive configuration for custom base fuel composition.
    
    Returns:
        tuple: (smiles_list, mole_fractions_list) or (None, None) if cancelled
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    print("\n" + "="*70)
    print("CUSTOM BASE FUEL CONFIGURATION")
    print("="*70)

    
    # Get number of components
    while True:
        num_comp_str = input("\nHow many components in your base fuel? (2-10): ").strip()
        if not num_comp_str.isdigit():
            print("❌ Please enter a positive integer.")
            continue
        
        num_comp = int(num_comp_str)
        if num_comp < 2:
            print("❌ Base fuel must have at least 2 components.")
            continue
        if num_comp > 10:
            print("⚠ Warning: Using many components may slow down predictions.")
            print("   Model supports max 11 total (base + additive).")
            confirm = input("   Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        break
    
    print(f"\n✓ Configuring {num_comp} components")
    print("-" * 70)
    
    # Collect components
    smiles_list = []
    fraction_list = []
    total_fraction = 0.0
    
    for i in range(num_comp):
        print(f"\nBase Fuel Component {i+1}/{num_comp}:")
        print("-" * 40)
        
        # Get SMILES
        while True:
            smiles = input("Enter SMILES: ").strip()
            
            if not smiles:
                print(" SMILES cannot be empty.")
                continue
            
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f" Invalid SMILES: '{smiles}'")
                print("     Please enter a valid SMILES string.")
                continue
            
            # Show molecule info
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            mol_weight = Descriptors.MolWt(mol)
            print(f"  ✓ Valid molecule: {formula} (MW: {mol_weight:.2f} g/mol)")
            break
        
        # Get mole fraction
        while True:
            # For last component, can auto-calculate or allow input
            if i == num_comp - 1:
                remaining = 1.0 - total_fraction
                print(f"  Remaining mole fraction: {remaining:.4f}")
                auto = input("  Use remaining fraction? (y/n): ").strip().lower()
                if auto == 'y':
                    mole_frac = remaining
                    print(f"  ✓ Mole fraction: {mole_frac:.4f}")
                    break
            
            frac_input = input(f"  Enter mole fraction (0-1, remaining: {1.0 - total_fraction:.4f}): ").strip()
            
            try:
                mole_frac = float(frac_input)
            except ValueError:
                print(" Invalid input. Please enter a number between 0 and 1.")
                continue
            
            if mole_frac <= 0 or mole_frac > 1:
                print(" Mole fraction must be between 0 and 1 (exclusive of 0).")
                continue
            
            if total_fraction + mole_frac > 1.0001:  # Small tolerance
                print(f" Total would exceed 1.0 (current total: {total_fraction:.4f})")
                continue
            
            break
        
        smiles_list.append(smiles)
        fraction_list.append(mole_frac)
        total_fraction += mole_frac
        
        print(f"  ✓ Component added (Total fraction: {total_fraction:.4f})")
    
    # Normalize fractions to exactly 1.0
    if abs(total_fraction - 1.0) > 1e-6:
        print(f"\n⚠ Normalizing mole fractions (sum was {total_fraction:.6f})")
        fraction_list = [f / total_fraction for f in fraction_list]
    
    # Display summary
    print("\n" + "="*70)
    print("CUSTOM BASE FUEL SUMMARY")
    print("="*70)
    print(f"\n{'#':<5} {'Formula':<15} {'SMILES':<35} {'Mole Frac':<10}")
    print("-" * 70)
    
    for i, (smiles, frac) in enumerate(zip(smiles_list, fraction_list), 1):
        mol = Chem.MolFromSmiles(smiles)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        smiles_display = smiles if len(smiles) <= 33 else smiles[:30] + "..."
        print(f"{i:<5} {formula:<15} {smiles_display:<35} {frac:.4f}")
    
    print("-" * 70)
    print(f"{'TOTAL':<5} {'':<15} {'':<35} {sum(fraction_list):.4f}")
    print("="*70)
    
    # Confirm
    confirm = input("\nUse this custom base fuel? (y/n): ").strip().lower()
    if confirm != 'y':
        print("⚠ Cancelled. Using fossil diesel as default.")
        return None, None
    
    return smiles_list, fraction_list


def get_mixture_config() -> EvolutionConfig:
    """Get config for mixture optimization."""
    
    print("\n✓ Mode: Mixture/Blend Optimization")
    print("\nOptimization Mode:")
    print("1. Target a specific CN value (minimize error from target)")
    print("2. Maximize CN (find highest possible CN)")
    mode = input("Select mode (1 or 2): ").strip()
    maximize_cn = (mode == "2")
    while mode not in ["1", "2"]:
        print("Invalid selection. Please choose 1 or 2.")
        mode = input("Select mode (1 or 2): ").strip()
        maximize_cn = (mode == "2")
    if maximize_cn:
        print("\n✓ Mode: Maximize Cetane Number")
        target_dcn = 100.0
    else:
        print("\n✓ Mode: Target Cetane Number")
        while True:
            try:
                target_dcn = float(input("Enter target mixture DCN (30-80): ").strip())
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    
    # Base fuel selection
    print("\nBase Fuel Selection:")
    print("1. Fossil diesel (typical petroleum diesel)")
    print("2. Biodiesel (FAME)")
    print("3. Custom (enter your own composition)")
    print("   → Define your own SMILES and mole fractions")
    
    base_fuel_choice = input("\nSelect base fuel (1-3): ").strip()
    
    if base_fuel_choice == "1":
        base_fuel_type = "fossil_diesel"
        base_fuel_smiles = None
        base_fuel_fractions = None
        print("✓ Using fossil diesel as base fuel")
        print("  Components: C16, C17, C18 n-alkanes + branched + aromatics + cyclic")
    
    elif base_fuel_choice == "2":
        base_fuel_type = "biodiesel"
        base_fuel_smiles = None
        base_fuel_fractions = None
        print("✓ Using biodiesel as base fuel")
        print("  Components: Stearate, Oleate, Linoleate, Palmitate (methyl esters)")
    
    elif base_fuel_choice == "3":
        base_fuel_smiles, base_fuel_fractions = get_custom_base_fuel()
        
        # If user cancelled, use default
        if base_fuel_smiles is None:
            base_fuel_type = "fossil_diesel"
            print("  → Defaulting to fossil diesel")
        else:
            base_fuel_type = "custom"
            print(f"\n✓ Custom base fuel configured")
            print(f"  Total components: {len(base_fuel_smiles)}")
            print(f"  Total mole fraction: {sum(base_fuel_fractions):.4f}")
    
    else:
        print(f"⚠ Invalid choice '{base_fuel_choice}', using fossil diesel as default")
        base_fuel_type = "fossil_diesel"
        base_fuel_smiles = None
        base_fuel_fractions = None
    

    
    while True:
        try:
            additive_fraction_str = input("\nEnter additive molar fraction (0.01-0.30): ").strip()
            additive_fraction = float(additive_fraction_str)
            
            if additive_fraction <= 0 or additive_fraction >= 1:
                print("❌ Additive fraction must be between 0 and 1 (exclusive).")
                continue
            
            break
        except ValueError:
            print("❌ Invalid input. Please enter a number (e.g., 0.1 for 10%).")
    
    # Summary
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY:")
    print("="*70)
    print(f"  • Mode: Mixture Optimization")
    print(f"  • Target Mixture DCN: {target_dcn}")
    print(f"  • Base Fuel Type: {base_fuel_type}")
    if base_fuel_smiles:
        print(f"  • Base Fuel Components: {len(base_fuel_smiles)}")
    print(f"  • Additive Fraction: {additive_fraction * 100:.1f}%")
    print(f"  • Base Fuel Fraction: {(1-additive_fraction) * 100:.1f}%")
    print("="*70)

    # Create mixture config
    mixture_cfg = MixtureConfig(
        additive_fraction=additive_fraction,
        base_fuel_fraction=1.0 - additive_fraction,
        base_fuel_type=base_fuel_type,
        base_fuel_smiles=base_fuel_smiles,
        base_fuel_mole_fractions=base_fuel_fractions,
        target_mixture_dcn=target_dcn
    )
    
    return EvolutionConfig(
        target_cn=target_dcn,  
        maximize_cn=maximize_cn,
        minimize_ysi=False,  # TODO: Add mixture YSI support later
        mixture_mode=True,
        mixture_config=mixture_cfg
    )

