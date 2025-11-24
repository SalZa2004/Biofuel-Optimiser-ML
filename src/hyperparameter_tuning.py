"""
Optuna hyperparameter tuning for LightGBM CN prediction.
"""
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from feature_selection import get_selected_data

# Load data
X_train, y_train = get_selected_data()

def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 25, 80),        # Higher
        "max_depth": trial.suggest_int("max_depth", 5, 12),           # Higher
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "n_estimators": 1000,
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),  # Lower
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 0.5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1.5, log=True),
        "random_state": 42,
        "verbose": -1,
    }
    # ... rest same
    
    model = LGBMRegressor(**params)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in cv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Early stopping - stops when validation score stops improving
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        pred = model.predict(X_val)
        rmse_scores.append(np.sqrt(np.mean((y_val - pred) ** 2)))
    
    return np.mean(rmse_scores)


if __name__ == "__main__":
    # Create study
    study = optuna.create_study(
        direction="minimize",  # Minimize RMSE
        sampler=TPESampler(seed=42),
        study_name="lgbm_cn_tuning"
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=100,          # Number of trials
        show_progress_bar=True,
        n_jobs=1               # Optuna parallelism (keep 1, CV is parallel)
    )
    
    # Results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best RMSE (CV): {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best params
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)
    
    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["verbose"] = -1
    best_params["n_jobs"] = -1
    
    final_model = LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    # Show top 5 trials
    print("\n" + "="*60)
    print("TOP 5 TRIALS")
    print("="*60)
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value").head(5)
    print(trials_df[["number", "value", "params_num_leaves", "params_max_depth", 
                     "params_n_estimators", "params_learning_rate"]].to_string())
    
    # Optional: Save study for later analysis
    # import joblib
    # joblib.dump(study, "optuna_study.pkl")

