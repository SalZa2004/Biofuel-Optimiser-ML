import pandas as pd
from .config import all_training_fames, main_fames, minor_fames, fame_groups


def _prompt_fame_group(fames: list, highlight: set) -> dict:
    """Prompt the user for FAME values in a group; Enter accepts 0."""
    data = {}
    for fame in fames:
        marker = " *" if fame in highlight else "  "
        raw = input(f"{marker}{fame} [0]: ").strip()
        try:
            data[fame] = float(raw) if raw else 0.0
        except ValueError:
            print(f"   Invalid value — using 0.")
            data[fame] = 0.0
    return data


def _get_single_sample() -> pd.DataFrame:
    print("\n" + "-" * 70)
    print("Enter FAME composition (% by weight).  Press Enter to use 0.")
    print("  * marks FAMEs most commonly used by the model.")
    print("-" * 70)

    main_set = set(main_fames)
    data = {}

    for group_name, fames in fame_groups.items():
        print(f"\n  [{group_name}]")
        data.update(_prompt_fame_group(fames, main_set))

    total = sum(data.values())
    print(f"\n  Total entered: {total:.2f}%")
    if total == 0:
        print("  Warning: all values are 0 — this sample will be rejected as invalid.")

    return pd.DataFrame([data])


def _get_csv() -> pd.DataFrame:
    print("\n" + "-" * 70)
    print("CSV format: one row per sample, columns named by FAME (e.g. C18:1).")
    print("Missing FAME columns are filled with 0 automatically.")
    print("-" * 70)

    while True:
        path = input("\nEnter path to CSV file: ").strip()
        try:
            df = pd.read_csv(path)
            print(f"  Loaded {len(df)} sample(s) from '{path}'")
            return df
        except FileNotFoundError:
            print(f"  File not found: '{path}'. Please try again.")
        except Exception as e:
            print(f"  Error loading file: {e}")


def get_user_config() -> dict:
    """Collect prediction mode and input data from the user."""
    print("\n" + "=" * 70)
    print("BIODIESEL CN PREDICTOR — Configuration")
    print("=" * 70)

    mode = input("\nSelect prediction mode (1: Single sample, 2: CSV batch): ").strip()
    while mode not in {"1", "2"}:
        print("Invalid selection. Please choose 1 or 2.")
        mode = input("Select prediction mode (1: Single sample, 2: CSV batch): ").strip()

    if mode == "1":
        return {"mode": "single", "df": _get_single_sample()}
    else:
        return {"mode": "csv", "df": _get_csv()}
