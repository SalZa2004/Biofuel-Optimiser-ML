from applications.mixture_predictor.cli import get_user_config
from applications.mixture_predictor.results import display_mixture_summary, save_results, predict_dcn
from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor


def main():
    """Main CLI loop."""
    print("\n" + "="*70)
    print("        MIXTURE DERIVED CETANE NUMBER (DCN) PREDICTOR")
    print("="*70)
    print("\nPredict the cetane number of fuel mixtures from molecular structure.")
    print("Enter SMILES strings and mole fractions for each component.")
    
    while True:
        try:
            # Get user configuration
            components = get_user_config()
            
            # Display summary
            display_mixture_summary(components)
            
            # Confirm before prediction
            confirm = input("\nContinue? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Prediction cancelled.")
                continue
            
            # Make prediction
            dcn = predict_dcn(components)
            
            if dcn is not None:
                # Save results
                save_results(components, dcn)
            break
            
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user. Exiting...")
            print("="*70 + "\n")
            break
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("\nTry again? (y/n): ", end='')
            retry = input().strip().lower()
            if retry != 'y':
                break


if __name__ == "__main__":
    main()







