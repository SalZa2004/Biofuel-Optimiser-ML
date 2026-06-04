import pandas as pd
from .config import all_training_fames, features


def ensure_fame_columns(df):
    """
    Ensure all training FAME columns exist in the input DataFrame.
    Missing FAME columns are added with 0; blank/non-numeric values are coerced to 0.
    """
    df = df.copy()

    for col in all_training_fames:
        if col not in df.columns:
            df[col] = 0.0

    df[all_training_fames] = (
        df[all_training_fames]
        .replace(r"^\s*$", 0, regex=True)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    return df


def validate_biodiesel(df):
    """
    Apply validation rules:
    - Total composition between 95–105%
    - Individual FAME values between 0–100%
    - C18:3 ≤ 12%
    - C18:2 < 70%
    """
    df = df.copy()

    df["Total %"] = df[all_training_fames].sum(axis=1)

    df["valid_total"] = df["Total %"].between(95, 105)
    df["valid_range"] = (
        (df[all_training_fames] >= 0).all(axis=1)
        & (df[all_training_fames] <= 100).all(axis=1)
    )
    df["valid_C18_3"] = df["C18:3"] <= 12
    df["valid_C18_2"] = df["C18:2"] < 70

    df["is_valid_biodiesel"] = (
        df["valid_total"]
        & df["valid_range"]
        & df["valid_C18_3"]
        & df["valid_C18_2"]
    )

    return df


def normalize_fames(df):
    """
    Normalize FAME composition so that total = 100%.
    Records the original total and whether normalization was applied.
    """
    df = df.copy()

    df["Original Total %"] = df["Total %"]
    df["was_normalized"] = ~df["Original Total %"].round(6).eq(100)

    df[all_training_fames] = (
        df[all_training_fames]
        .div(df["Original Total %"], axis=0)
        * 100
    )

    df["Total %"] = df[all_training_fames].sum(axis=1)

    return df


def engineer_features(df):
    """
    Generate model input features from FAME composition:
    - Average carbon number and double bonds
    - SFA, MUFA, PUFA fractions
    - Approximate molecular weight
    """
    df = df.copy()

    carbon_map = {col: int(col.split(":")[0][1:]) for col in all_training_fames}
    db_map = {col: int(col.split(":")[1]) for col in all_training_fames}

    df["avg_carbon"] = sum(df[c] * carbon_map[c] for c in all_training_fames) / 100
    df["avg_double_bonds"] = sum(df[c] * db_map[c] for c in all_training_fames) / 100

    df["SFA"] = sum(df[c] for c in all_training_fames if db_map[c] == 0)
    df["MUFA"] = sum(df[c] for c in all_training_fames if db_map[c] == 1)
    df["PUFA"] = sum(df[c] for c in all_training_fames if db_map[c] >= 2)

    df["avg_mw"] = df["avg_carbon"] * 14 + 46

    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    return df


def predict_cn(df, model):
    df = df.copy()
    df["CN_pred"] = model.predict(df[features].values)
    return df


def add_ood_flags(df, ood_stats):
    """
    Flag samples outside the feature ranges seen during training.
    """
    df = df.copy()

    ood_features = ood_stats["features"]
    mins = ood_stats["feature_min"]
    maxs = ood_stats["feature_max"]

    X = df[ood_features].values

    below_min = X < mins
    above_max = X > maxs

    df["ood_flag"] = (below_min | above_max).any(axis=1)

    return df


def run_prediction_pipeline(df, model, ood_stats=None):
    """
    Full prediction workflow:
    1. Ensure required FAME columns exist
    2. Validate biodiesel composition
    3. Split valid/invalid samples
    4. Normalize valid samples
    5. Engineer features
    6. OOD detection (if ood_stats provided)
    7. Predict cetane number

    Returns:
        valid_df   – processed and predicted samples
        invalid_df – rejected samples with validation flags
    """
    df = ensure_fame_columns(df)
    df = validate_biodiesel(df)

    valid_df = df[df["is_valid_biodiesel"]].copy()
    invalid_df = df[~df["is_valid_biodiesel"]].copy()

    if len(valid_df) > 0:
        valid_df = normalize_fames(valid_df)
        valid_df = engineer_features(valid_df)

        if ood_stats is not None:
            valid_df = add_ood_flags(valid_df, ood_stats)

        valid_df = predict_cn(valid_df, model)

    return valid_df, invalid_df
