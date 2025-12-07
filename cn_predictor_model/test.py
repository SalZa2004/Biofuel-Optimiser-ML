"""
Test the trained CN prediction model on a held-out test set.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from cn_model.model import CetanePredictor
import joblib
from train import FeatureSelector

import os

# 1. Location of this file (test.py)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Project root: one level up
PROJECT_ROOT = os.path.dirname(TEST_DIR)

# 3. Build DB path
DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "database_main.db")

print("DB_PATH:", DB_PATH)
print("DB_EXISTS:", os.path.exists(DB_PATH))

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(PROJECT_ROOT,"cn-predictor-model", "cn_model", "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")

def load_and_split_data(test_size=0.2, random_state=42):
    """
    Load data from database and split into train/test.
    Returns the test set for evaluation.
    """
    print("="*70)
    print("LOADING DATA FROM DATABASE")
    print("="*70)
    
    # Load data
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        F.Fuel_Name,
        F.SMILES,
        T.Standardised_DCN AS cn
    FROM FUEL F
    LEFT JOIN TARGET T ON F.fuel_id = T.fuel_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Clean data
    df.dropna(subset=["cn", "SMILES"], inplace=True)
    
    # Remove outliers (same as training)
    df = df[(df["cn"] >= 0) & (df["cn"] <= 150)]
    
    print(f"Total samples: {len(df)}")
    print(f"CN range: [{df['cn'].min():.1f}, {df['cn'].max():.1f}]")
    
    # Split
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}")
    
    return train_df, test_df


def evaluate_model(predictor, test_df):
    """
    Evaluate the model on test set.
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = predictor.predict(test_df["SMILES"].tolist())
    
    # Handle cases where some molecules might be invalid
    if len(predictions) != len(test_df):
        print(f"⚠ Warning: {len(test_df) - len(predictions)} molecules failed featurization")
        # Get valid predictions with details
        results = predictor.predict_with_details(test_df["SMILES"].tolist())
        results = results.merge(test_df[["SMILES", "cn"]], on="SMILES", how="left")
        results = results[results["Valid"] == True]
        
        y_true = results["cn"].values
        y_pred = results["Predicted_CN"].values
    else:
        y_true = test_df["cn"].values
        y_pred = np.array(predictions)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate relative errors
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"R²:    {r2:.4f}")
    print(f"MAPE:  {mape:.2f}%")
    
    return y_true, y_pred, {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def evaluate_training_set(predictor, train_df):
    """
    Evaluate the model on training set to check for overfitting.
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TRAINING SET")
    print("="*70)
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = predictor.predict(train_df["SMILES"].tolist())
    
    # Handle cases where some molecules might be invalid
    if len(predictions) != len(train_df):
        results = predictor.predict_with_details(train_df["SMILES"].tolist())
        results = results.merge(train_df[["SMILES", "cn"]], on="SMILES", how="left")
        results = results[results["Valid"] == True]
        
        y_true = results["cn"].values
        y_pred = results["Predicted_CN"].values
    else:
        y_true = train_df["cn"].values
        y_pred = np.array(predictions)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("\n" + "="*70)
    print("TRAINING SET RESULTS")
    print("="*70)
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"R²:    {r2:.4f}")
    print(f"MAPE:  {mape:.2f}%")
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def plot_results(y_true, y_pred, save_path="evaluation_plots.png"):
    """
    Create visualization of predictions vs actual values.
    """
    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Scatter plot: Predicted vs Actual
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=30)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    ax1.set_xlabel('Actual CN', fontsize=12)
    ax1.set_ylabel('Predicted CN', fontsize=12)
    ax1.set_title('Predicted vs Actual Cetane Number', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residual plot
    ax2 = axes[1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted CN', fontsize=12)
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax3 = axes[2]
    ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax3.text(0.05, 0.95, f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}', 
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to {save_path}")
    
    return fig


def analyze_errors(test_df, y_true, y_pred, top_n=10):
    """
    Analyze worst predictions.
    """
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    
    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)
    
    # Get indices of worst predictions
    worst_indices = np.argsort(errors)[-top_n:][::-1]
    
    print(f"\nTop {top_n} worst predictions:")
    print("-" * 90)
    print(f"{'SMILES':<30} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    print("-" * 90)
    
    for idx in worst_indices:
        smiles = test_df.iloc[idx]["SMILES"]
        actual = y_true[idx]
        predicted = y_pred[idx]
        error = errors[idx]
        
        # Truncate SMILES if too long
        smiles_display = smiles[:27] + "..." if len(smiles) > 30 else smiles
        print(f"{smiles_display:<30} {actual:>10.2f} {predicted:>10.2f} {error:>10.2f}")


def save_predictions(test_df, y_true, y_pred, save_path="test_predictions.csv"):
    """
    Save all test predictions to CSV.
    """
    results_df = test_df.copy()
    results_df = results_df.reset_index(drop=True)
    results_df["Actual_CN"] = y_true
    results_df["Predicted_CN"] = y_pred
    results_df["Absolute_Error"] = np.abs(y_true - y_pred)
    results_df["Relative_Error_%"] = np.abs((y_true - y_pred) / y_true) * 100
    
    results_df.to_csv(save_path, index=False)
    print(f"\n✓ Predictions saved to {save_path}")


def main():
    """
    Main evaluation pipeline.
    """
    print("="*70)
    print("CETANE NUMBER MODEL EVALUATION")
    print("="*70)
    
    # Check if model exists
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print("\n❌ Error: Model not found!")
        print("Please train the model first: python train.py train")
        return
    
    # Load the trained model
    print("\nLoading trained model...")

    predictor = CetanePredictor()
    
    # Load and split data
    train_df, test_df = load_and_split_data(test_size=0.2, random_state=42)
    
    # Evaluate on training set
    train_metrics = evaluate_training_set(predictor, train_df)
    
    # Evaluate on test set
    y_true, y_pred, test_metrics = evaluate_model(predictor, test_df)
    
    # Check for overfitting
    print("\n" + "="*70)
    print("OVERFITTING CHECK")
    print("="*70)
    print(f"Train RMSE: {train_metrics['RMSE']:.4f}")
    print(f"Test RMSE:  {test_metrics['RMSE']:.4f}")
    print(f"Difference: {test_metrics['RMSE'] - train_metrics['RMSE']:.4f}")
    
    if test_metrics['RMSE'] > train_metrics['RMSE'] * 1.2:
        print("⚠ Warning: Model may be overfitting (test RMSE > 1.2x train RMSE)")
    else:
        print("✓ Model generalization looks good!")
    
    # Create visualizations
    plot_results(y_true, y_pred, save_path="cn-predictor-model/cn_model/evaluation_plots.png")
    
    # Error analysis
    analyze_errors(test_df, y_true, y_pred, top_n=10)
    
    # Save predictions
    save_predictions(test_df, y_true, y_pred, save_path="cn-predictor-model/cn_model/test_predictions.csv")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - cn_model/evaluation_plots.png")
    print("  - cn_model/test_predictions.csv")
    print("="*70)


if __name__ == "__main__":
    main()