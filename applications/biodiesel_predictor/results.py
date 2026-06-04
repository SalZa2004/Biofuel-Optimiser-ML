import pandas as pd
from .config import all_training_fames


def _input_fame_cols(df: pd.DataFrame) -> list:
    return [c for c in all_training_fames if c in df.columns and df[c].sum() > 0]


def display_results(valid_df: pd.DataFrame, invalid_df: pd.DataFrame, mode: str = "single"):
    print("\n" + "=" * 70)
    print("BIODIESEL CN PREDICTION RESULTS")
    print("=" * 70)

    if len(valid_df) > 0:
        print(f"\n  Valid samples: {len(valid_df)}")
        print("-" * 70)

        summary_cols = ["CN_pred"]
        if "ood_flag" in valid_df.columns:
            summary_cols.append("ood_flag")
        if "was_normalized" in valid_df.columns:
            summary_cols.append("was_normalized")
        if "Original Total %" in valid_df.columns:
            summary_cols.append("Original Total %")

        if mode == "single":
            row = valid_df.iloc[0]
            print(f"\n  Predicted Cetane Number: {row['CN_pred']:.2f}")
            if "ood_flag" in row.index:
                status = "Outside training range (reliability may be lower)" if row["ood_flag"] else "Within training range"
                print(f"  Distribution check:      {status}")
            if "was_normalized" in row.index and row["was_normalized"]:
                print(f"  Note: composition was normalized from {row['Original Total %']:.2f}% to 100%.")
        else:
            print(valid_df[summary_cols].to_string(index=True))

        if "ood_flag" in valid_df.columns:
            ood_count = int(valid_df["ood_flag"].sum())
            if ood_count > 0:
                print(f"\n  Warning: {ood_count} sample(s) fall outside the training distribution.")
                print("           CN predictions for these samples may be less reliable.")

    if len(invalid_df) > 0:
        print(f"\n  Invalid samples: {len(invalid_df)}")
        print("-" * 70)
        flag_cols = [c for c in ["Total %", "valid_total", "valid_range", "valid_C18_3", "valid_C18_2"] if c in invalid_df.columns]
        if flag_cols:
            print(invalid_df[flag_cols].to_string(index=True))
        print()
        print("  Validation rules:")
        print("    valid_total  — composition sum must be 95–105%")
        print("    valid_range  — each FAME must be 0–100%")
        print("    valid_C18_3  — C18:3 must be ≤ 12%")
        print("    valid_C18_2  — C18:2 must be < 70%")

    if len(valid_df) == 0 and len(invalid_df) == 0:
        print("\n  No samples processed.")

    print("\n" + "=" * 70)

    if len(valid_df) > 0:
        _save_results(valid_df, invalid_df)


def _save_results(valid_df: pd.DataFrame, invalid_df: pd.DataFrame):
    save = input("\nSave results to CSV? (y/n): ").strip().lower()
    if save != "y":
        return

    raw = input("Enter filename [default: biodiesel_predictions.csv]: ").strip()
    filename = raw if raw else "biodiesel_predictions.csv"
    if not filename.endswith(".csv"):
        filename += ".csv"

    try:
        valid_df.to_csv(filename, index=True)
        print(f"  Saved predictions to '{filename}'")

        if len(invalid_df) > 0:
            inv_filename = filename.replace(".csv", "_invalid.csv")
            invalid_df.to_csv(inv_filename, index=True)
            print(f"  Saved invalid samples to '{inv_filename}'")

    except Exception as e:
        print(f"  Failed to save: {e}")
