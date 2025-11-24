"""
Optuna hyperparameter tuning for ExtraTrees CN prediction.
ExtraTrees had the best baseline performance (Test RMSE: 9.04).
"""

import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from feature_selection import get_selected_data

# Load data
X_train, y_train = get_selected_data()

print(f"Data shape: {X_train.shape}")
print(f"Target range: {y_train.min():.1f} - {y_train.max():.1f}")

def objective(trial):
    """Optuna objective function for ExtraTrees."""
    
    params = {
    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
    "max_depth": trial.suggest_int("max_depth", 10, 30),        # Lower max
    "min_samples_split": trial.suggest_int("min_samples_split", 10, 40),  # Higher
    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),     # Higher
    "max_features": trial.suggest_float("max_features", 0.3, 0.8),
    "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.02),
    "bootstrap": True,  # Force bootstrap to reduce overfitting
    "random_state": 42,
    "n_jobs": -1
}
    
    model = ExtraTreesRegressor(**params)
    
    # 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv, 
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    
    # Return mean RMSE
    rmse = np.sqrt(-scores.mean())
    return rmse


if __name__ == "__main__":
    print("="*70)
    print("EXTRATREES HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name="extratrees_cn_tuning"
    )
    
    # Optimize
    print("\nRunning optimization...")
    study.optimize(
        objective,
        n_trials=100,
        show_progress_bar=True,
        n_jobs=1
    )
    
    # Results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Best CV RMSE: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best params
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)
    
    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    
    final_model = ExtraTreesRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    # Training performance
    train_pred = final_model.predict(X_train)
    train_rmse = np.sqrt(np.mean((y_train - train_pred)**2))
    train_mae = np.mean(np.abs(y_train - train_pred))
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train MAE:  {train_mae:.4f}")
    
    # Show top 5 trials
    print("\n" + "="*70)
    print("TOP 5 TRIALS")
    print("="*70)
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value").head(5)
    print(trials_df[["number", "value", "params_n_estimators", 
                     "params_max_depth", "params_min_samples_leaf"]].to_string(index=False))
    
    # Save study
    import joblib
    joblib.dump(study, "extratrees_study.pkl")
    joblib.dump(final_model, "extratrees_model.pkl")
    print("\nStudy and model saved!")
    
    # Feature importance
    print("\n" + "="*70)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*70)
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        print(f"{i:2d}. Feature {idx:3d}: {importances[idx]:.4f}")