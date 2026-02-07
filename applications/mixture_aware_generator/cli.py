"""
Updated CLI with Mixture Optimization Support
"""

from core.config import EvolutionConfig, MixtureConfig

def get_user_config() -> EvolutionConfig:
    """Interactive CLI to get configuration from user."""
    
    print("="*70)
    print("MOLECULAR EVOLUTION WITH GENETIC ALGORITHM")
    print("="*70)
    
    # Choose mode: pure component or mixture
    print("\nOptimization Target:")
    print("1. Pure component properties (original)")
    print("2. Mixture/blend properties (fuel additive optimization)")
    mode_choice = input("Select mode (1 or 2): ").strip()
    
    mixture_mode = (mode_choice == "2")
    
    if not mixture_mode:
        # Original pure component mode
        return get_pure_component_config()
    else:
        # New mixture mode
        return get_mixture_config()


def get_pure_component_config() -> EvolutionConfig:
    """Get config for pure component optimization (original)."""
    
    print("\n✓ Mode: Pure Component Optimization")
    
    print("\nCetane Number Optimization:")
    print("1. Target a specific CN value (minimize error from target)")
    print("2. Maximize CN (find highest possible CN)")
    opt_mode = input("Select mode (1 or 2): ").strip()
    
    maximize_cn = (opt_mode == "2")
    
    if maximize_cn:
        print("✓ Mode: Maximize Cetane Number")
        target_cn = 100.0  # Doesn't matter, but set it anyway
    else:
        print("✓ Mode: Target Cetane Number")
        target_cn = float(input("Enter target CN: ").strip())
    
    minimize_ysi_input = input("\nMinimize YSI (y/n): ").strip().lower()
    minimize_ysi = (minimize_ysi_input == 'y')
    
    # Summary
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY:")
    if maximize_cn:
        print("  • Mode: Maximize CN")
    else:
        print(f"  • Mode: Target CN = {target_cn}")
    print(f"  • Minimize YSI: {'Yes' if minimize_ysi else 'No'}")
    if minimize_ysi:
        print("  • Optimization: Multi-objective (CN + YSI)")
    else:
        print("  • Optimization: Single-objective (CN only)")
    print("="*70)
    
    return EvolutionConfig(
        target_cn=target_cn,
        maximize_cn=maximize_cn,
        minimize_ysi=minimize_ysi,
        mixture_mode=False
    )


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
    additive_fraction = float(input("Enter additive fraction (e.g., 0.15 for 15%): ").strip())
    
    # Optimize blend ratio?
    optimize_ratio_input = input("Optimize blend ratio for each molecule? (y/n): ").strip().lower()
    optimize_ratio = (optimize_ratio_input == 'y')
    
    if optimize_ratio:
        min_fraction = float(input("  Min additive fraction (e.g., 0.05): ").strip())
        max_fraction = float(input("  Max additive fraction (e.g., 0.30): ").strip())
    else:
        min_fraction = additive_fraction
        max_fraction = additive_fraction
    
    # Summary
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY:")
    print(f"  • Mode: Mixture Optimization")
    print(f"  • Target Mixture DCN: {target_dcn}")
    print(f"  • Base Fuel: {base_fuel_type}")
    print(f"  • Additive Fraction: {additive_fraction * 100:.1f}%")
    print(f"  • Base Fuel Fraction: {(1-additive_fraction) * 100:.1f}%")
    if optimize_ratio:
        print(f"  • Optimize Ratio: Yes ({min_fraction*100:.1f}% - {max_fraction*100:.1f}%)")
    else:
        print(f"  • Optimize Ratio: No (fixed at {additive_fraction*100:.1f}%)")
    print("="*70)
    
    # Create mixture config
    mixture_cfg = MixtureConfig(
        additive_fraction=additive_fraction,
        base_fuel_fraction=1.0 - additive_fraction,
        base_fuel_type=base_fuel_type,
        base_fuel_smiles=base_fuel_smiles,
        base_fuel_mole_fractions=base_fuel_fractions,
        target_mixture_dcn=target_dcn,
        optimize_blend_ratio=optimize_ratio,
        min_additive_fraction=min_fraction,
        max_additive_fraction=max_fraction
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