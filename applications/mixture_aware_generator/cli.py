"""
Updated CLI with Mixture Optimization Support
"""
from core.config import EvolutionConfig, MixtureConfig
def get_user_config() -> EvolutionConfig:
    """Interactive CLI to get configuration from user."""
    
    print("="*70)
    print("MIXTURE-AWARE MOLECULE GENERATOR")
    print("="*70)
    
    return get_mixture_config()


def get_mixture_config() -> EvolutionConfig:
    """Get config for mixture optimization."""
    
    print("\n✓ Mode: Mixture/Blend Optimization")
    # Target mixture DCN
    print("\nTarget Mixture Properties:")
    target_dcn = float(input("Enter target mixture DCN: ").strip())
    
    # Base fuel selection
    print("\nBase Fuel Selection:")
    print("1. Fossil diesel (typical petroleum diesel)")
    print("2. Biodiesel (FAME)")
    print("3. Custom (enter your own composition)")
    base_fuel_choice = input("Select base fuel (1-3): ").strip()
    
    if base_fuel_choice == "1":
        base_fuel_type = "fossil_diesel"
        base_fuel_smiles = None
        base_fuel_fractions = None
        print("✓ Using fossil diesel as base fuel")
    elif base_fuel_choice == "2":
        base_fuel_type = "biodiesel"
        base_fuel_smiles = None
        base_fuel_fractions = None
        print("✓ Using biodiesel as base fuel")
    else:
        base_fuel_type = "custom"
        print("\nCustom base fuel not yet implemented in CLI")
        print("Using fossil diesel as default...")
        base_fuel_smiles = None
        base_fuel_fractions = None
    
    # Additive fraction
    print("\nBlend Ratio:")
    additive_fraction = float(input("Enter additive molar fraction").strip())
    
    
    # Summary
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY:")
    print(f"  • Mode: Mixture Optimization")
    print(f"  • Target Mixture DCN: {target_dcn}")
    print(f"  • Base Fuel: {base_fuel_type}")
    print(f"  • Additive Fraction: {additive_fraction * 100:.1f}%")
    print(f"  • Base Fuel Fraction: {(1-additive_fraction) * 100:.1f}%")

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
        target_cn=target_dcn,  # Re-use this for mixture DCN
        maximize_cn=False,
        minimize_ysi=False,  # TODO: Add mixture YSI support later
        mixture_mode=True,
        mixture_config=mixture_cfg
    )

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    config = get_user_config()
    
    print("\nGenerated config:")
    print(f"  mixture_mode: {config.mixture_mode}")
    if config.mixture_mode:
        print(f"  target_mixture_dcn: {config.mixture_config.target_mixture_dcn}")
        print(f"  additive_fraction: {config.mixture_config.additive_fraction}")
        print(f"  base_fuel_type: {config.mixture_config.base_fuel_type}")