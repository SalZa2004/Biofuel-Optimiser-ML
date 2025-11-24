"""
Compare tuned ExtraTrees and XGBoost models on test set.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from feature_selection import get_selected_data

# Load data
X, y = get_selected_data()

# Split into train/test (same as baseline evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("="*70)
print("MODEL COMPARISON ON TEST SET")
print("="*70)
print(f"Train samples: {len(y_train)}")
print(f"Test samples:  {len(y_test)}")

# Load models
try:
    et_model = joblib.load("extratrees_model.pkl")
    et_available = True
except:
    et_available = False
    print("\nExtraTrees model not found. Run tune_extratrees.py first.")

try:
    xgb_model = joblib.load("xgboost_model.pkl")
    xgb_available = True
except:
    xgb_available = False
    print("\nXGBoost model not found. Run tune_xgboost.py first.")

if not et_available and not xgb_available:
    print("\nNo models found to compare!")
    exit()

results = []

# Evaluate ExtraTrees
if et_available:
    print("\n" + "-"*70)
    print("EXTRATREES")
    print("-"*70)
    
    et_train_pred = et_model.predict(X_train)
    et_test_pred = et_model.predict(X_test)
    
    et_results = {
        'Model': 'ExtraTrees',
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, et_train_pred)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, et_test_pred)),
        'Train_MAE': mean_absolute_error(y_train, et_train_pred),
        'Test_MAE': mean_absolute_error(y_test, et_test_pred),
        'Train_R2': r2_score(y_train, et_train_pred),
        'Test_R2': r2_score(y_test, et_test_pred)
    }
    
    et_results['Gap'] = et_results['Test_RMSE'] - et_results['Train_RMSE']
    et_results['Pct_Error'] = (et_results['Test_RMSE'] / y_test.mean()) * 100
    
    results.append(et_results)
    
    print(f"Train RMSE: {et_results['Train_RMSE']:.4f}")
    print(f"Test RMSE:  {et_results['Test_RMSE']:.4f}")
    print(f"Train MAE:  {et_results['Train_MAE']:.4f}")
    print(f"Test MAE:   {et_results['Test_MAE']:.4f}")
    print(f"Test R²:    {et_results['Test_R2']:.4f}")
    print(f"Gap:        {et_results['Gap']:.4f}")
    print(f"% Error:    {et_results['Pct_Error']:.2f}%")

# Evaluate XGBoost
if xgb_available:
    print("\n" + "-"*70)
    print("XGBOOST")
    print("-"*70)
    
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)
    
    xgb_results = {
        'Model': 'XGBoost',
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, xgb_train_pred)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, xgb_test_pred)),
        'Train_MAE': mean_absolute_error(y_train, xgb_train_pred),
        'Test_MAE': mean_absolute_error(y_test, xgb_test_pred),
        'Train_R2': r2_score(y_train, xgb_train_pred),
        'Test_R2': r2_score(y_test, xgb_test_pred)
    }
    
    xgb_results['Gap'] = xgb_results['Test_RMSE'] - xgb_results['Train_RMSE']
    xgb_results['Pct_Error'] = (xgb_results['Test_RMSE'] / y_test.mean()) * 100
    
    results.append(xgb_results)
    
    print(f"Train RMSE: {xgb_results['Train_RMSE']:.4f}")
    print(f"Test RMSE:  {xgb_results['Test_RMSE']:.4f}")
    print(f"Train MAE:  {xgb_results['Train_MAE']:.4f}")
    print(f"Test MAE:   {xgb_results['Test_MAE']:.4f}")
    print(f"Test R²:    {xgb_results['Test_R2']:.4f}")
    print(f"Gap:        {xgb_results['Gap']:.4f}")
    print(f"% Error:    {xgb_results['Pct_Error']:.2f}%")

# Summary table
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Winner
if len(results) == 2:
    winner = results_df.loc[results_df['Test_RMSE'].idxmin(), 'Model']
    print(f"\n🏆 Best model: {winner}")

# Visualization
if et_available and xgb_available:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ExtraTrees predictions
    axes[0].scatter(y_test, et_test_pred, alpha=0.5, s=20)
    axes[0].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('True CN')
    axes[0].set_ylabel('Predicted CN')
    axes[0].set_title(f'ExtraTrees (RMSE: {et_results["Test_RMSE"]:.2f})')
    axes[0].grid(True, alpha=0.3)
    
    # XGBoost predictions
    axes[1].scatter(y_test, xgb_test_pred, alpha=0.5, s=20)
    axes[1].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('True CN')
    axes[1].set_ylabel('Predicted CN')
    axes[1].set_title(f'XGBoost (RMSE: {xgb_results["Test_RMSE"]:.2f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    print("\nPlot saved as 'model_comparison.png'")
    plt.show()

# Ensemble prediction (average of both)
if et_available and xgb_available:
    print("\n" + "="*70)
    print("ENSEMBLE (Average of both models)")
    print("="*70)
    
    ensemble_test_pred = (et_test_pred + xgb_test_pred) / 2
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
    ensemble_mae = mean_absolute_error(y_test, ensemble_test_pred)
    ensemble_r2 = r2_score(y_test, ensemble_test_pred)
    ensemble_pct = (ensemble_rmse / y_test.mean()) * 100
    
    print(f"Test RMSE:  {ensemble_rmse:.4f}")
    print(f"Test MAE:   {ensemble_mae:.4f}")
    print(f"Test R²:    {ensemble_r2:.4f}")
    print(f"% Error:    {ensemble_pct:.2f}%")
    
    if ensemble_rmse < min(et_results['Test_RMSE'], xgb_results['Test_RMSE']):
        print("\n✨ Ensemble is better than individual models!")