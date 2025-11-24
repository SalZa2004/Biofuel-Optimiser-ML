"""
Optuna hyperparameter tuning for ExtraTrees CN prediction.
"""
from sklearn.ensemble import ExtraTreesRegressor
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from feature_selection import get_selected_data

# Load data
X_train, y_train = get_selected_data()

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'random_state': 42,
        'n_jobs': -1  # Use all cores
    }
    
    model = ExtraTreesRegressor(**params)
    
    # Use cross_val_score for simpler implementation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv, 
        scoring='neg_root_mean_squared_error',  # Negative RMSE (maximize)
        n_jobs=1  # Keep as 1 to avoid conflicts with Optuna
    )
    
    # Return positive RMSE (since we're minimizing)
    return -scores.mean()

if __name__ == "__main__":
    # Create study
    study = optuna.create_study(
        direction="minimize",  # Minimize RMSE
        sampler=TPESampler(seed=42),
        study_name="extra_trees_cn_tuning"
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=100,
        show_progress_bar=True,
        n_jobs=1  # Optuna parallelism
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
    
    best_params = study.best_params.copy()
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    
    final_model = ExtraTreesRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
