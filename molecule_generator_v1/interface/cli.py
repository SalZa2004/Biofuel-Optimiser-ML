from config import EvolutionConfig


def get_user_config() -> EvolutionConfig:
    """Get configuration from user input."""
    print("\n" + "="*70)
    print("MOLECULAR EVOLUTION WITH GENETIC ALGORITHM")
    print("="*70)
    
    # Choose optimization mode
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
        target = 100.0  # Dummy target, not used in maximize mode
    else:
        print("\n✓ Mode: Target Cetane Number")
        while True:
            target = float(input("Enter target CN: ") or "50")
            if target > 40:
                break
            print("⚠️  Target CN is too low, optimization may be challenging.")
            print("Consider using a higher target CN for better results.\n")
    
    # Ask about YSI
    minimize_ysi = input("\nMinimize YSI (y/n): ").strip().lower() in ['y', 'yes']
    
    # Print configuration summary
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY:")
    print(f"  • Mode: {'Maximize CN' if maximize_cn else f'Target CN = {target}'}")
    print(f"  • Minimize YSI: {'Yes' if minimize_ysi else 'No'}")
    print(f"  • Optimization: {'Multi-objective (CN + YSI)' if minimize_ysi else 'Single-objective (CN only)'}")
    print("="*70 + "\n")
    
    return EvolutionConfig(target_cn=target, maximize_cn=maximize_cn, minimize_ysi=minimize_ysi)