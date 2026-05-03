from applications.binary_blending_tool.cli import get_user_config
from applications.binary_blending_tool.results import (
    run_blend_sweep,
    display_sweep_results,
    save_results,
)


def main():
    print("\n" + "=" * 70)
    print("        BINARY BLENDING TOOL")
    print("=" * 70)
    print("\nSweep an additive's mole fraction and see how DCN and YSI respond.")

    while True:
        try:
            config = get_user_config()

            confirm = input("\nRun sweep? (y/n): ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                continue

            result = run_blend_sweep(**config)

            display_sweep_results(result)

            save = input("\nSave results to file? (y/n): ").strip().lower()
            if save == "y":
                raw = input("Filename prefix (no extension) [default: auto]: ").strip()
                filename = raw if raw else None
                save_results(result, filename)

            break

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user. Exiting...")
            print("=" * 70 + "\n")
            break

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            retry = input("\nTry again? (y/n): ").strip().lower()
            if retry != "y":
                break


if __name__ == "__main__":
    main()
