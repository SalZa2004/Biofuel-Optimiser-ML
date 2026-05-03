from applications.mixture_predictor.cli import get_user_config
from applications.mixture_predictor.results import (
    display_mixture_summary,
    predict_mixture_properties,
    save_results,
)


def main():
    """Main CLI loop."""
    print("\n" + "="*70)
    print("        MIXTURE PROPERTY PREDICTOR")
    print("="*70)
    print("\nPredict DCN, YSI, boiling point, and density of fuel mixtures.")
    print("Enter SMILES strings and mole fractions for each component.")

    while True:
        try:
            components = get_user_config()

            display_mixture_summary(components)

            confirm = input("\nContinue? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Prediction cancelled.")
                continue

            predictions = predict_mixture_properties(components)

            if predictions is not None:
                save_results(components, predictions)
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







