"""
Optuna hyperparameter tuning for XGBoost CN prediction.
XGBoost had second-best baseline performance (Test RMSE: 9.12).
"""

import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from feature_selection import get_selected_data

# Load data
X_train, y_train = get_selected_data()

print(f"Data shape: {X_train.shape}")
print(f"Target range: {y_train.min():.1f} - {y_train.max():.1f}")

def objective(trial):
    """Optuna objective function for XGBoost."""
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist"  # Faster training
    }
    
    model = XGBRegressor(**params)
    
    # 5-fold cross-validation with early stopping
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in cv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        pred = model.predict(X_val)
        rmse_scores.append(np.sqrt(np.mean((y_val - pred)**2)))
    
    return np.mean(rmse_scores)


if __name__ == "__main__":
    print("="*70)
    print("XGBOOST HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name="xgboost_cn_tuning"
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
    best_params["tree_method"] = "hist"
    
    final_model = XGBRegressor(**best_params)
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
                     "params_max_depth", "params_learning_rate"]].to_string(index=False))
    
    # Save study and model
    import joblib
    joblib.dump(study, "xgboost_study.pkl")
    joblib.dump(final_model, "xgboost_model.pkl")
    print("\nStudy and model saved!")
    
    # Feature importance
    print("\n" + "="*70)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*70)
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        print(f"{i:2d}. Feature {idx:3d}: {importances[idx]:.4f}")