import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import pickle
from huggingface_hub import hf_hub_download

from .cli import get_user_config
from .pipeline import run_prediction_pipeline
from .results import display_results

_HF_REPO = "mrashid26/Biodiesel_CN_Predictor"


def _load_pkl(filename: str):
    path = hf_hub_download(repo_id=_HF_REPO, filename=filename, repo_type="space")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        import joblib
        return joblib.load(path)


def main():
    print("\n" + "=" * 70)
    print("        BIODIESEL CN PREDICTOR")
    print("=" * 70)
    print("\nPredict cetane number of biodiesel from FAME composition.")

    print("\nLoading model...")
    model = _load_pkl("src/biodiesel_cn_model.pkl")
    ood_stats = _load_pkl("src/ood_stats.pkl")
    print("Model loaded.\n")

    while True:
        try:
            config = get_user_config()

            valid_df, invalid_df = run_prediction_pipeline(
                config["df"], model, ood_stats
            )

            display_results(valid_df, invalid_df, config["mode"])

            again = input("\nRun another prediction? (y/n): ").strip().lower()
            if again != "y":
                print("\n" + "=" * 70 + "\n")
                break

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting.")
            break

        except Exception as e:
            print(f"\n  Error: {e}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != "y":
                break


if __name__ == "__main__":
    main()
