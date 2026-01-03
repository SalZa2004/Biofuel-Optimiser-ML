"""
Enhanced test script with comprehensive model evaluation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from density_model.model import DensityPredictor
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import sys
from train import FeatureSelector
# Setup paths
PROJECT_ROOT = os.getcwd()
sys.path.append(PROJECT_ROOT)

def load_and_split_data(test_size=0.2, random_state=42):
    """Load data and split into train/test."""
    df = pd.read_csv('density.csv')
    df.dropna(subset=["density", "SMILES"], inplace=True)
    
    print(f"Total samples: {len(df)}")
    
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}")
    
    return train_df, test_df


def calculate_molecular_properties(smiles_list):
    """Calculate molecular properties for analysis."""
    properties = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            properties.append({
                'MolWt': np.nan,
                'NumHeavyAtoms': np.nan,
                'NumRotatableBonds': np.nan,
                'NumRings': np.nan,
                'NumAromaticRings': np.nan,
                'TPSA': np.nan,
                'logP': np.nan
            })
            continue
        
        properties.append({
            'MolWt': Descriptors.MolWt(mol),
            'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
            'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'NumRings': rdMolDescriptors.CalcNumRings(mol),
            'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'TPSA': rdMolDescriptors.CalcTPSA(mol),
            'logP': Descriptors.MolLogP(mol)
        })
    
    return pd.DataFrame(properties)


def evaluate_model(predictor, test_df):
    """Evaluate the model on test set."""
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    predictions = predictor.predict(test_df["SMILES"].tolist())
    
    if len(predictions) != len(test_df):
        results = predictor.predict_with_details(test_df["SMILES"].tolist())
        results = results.merge(test_df[["SMILES", "density"]], on="SMILES", how="left")
        results = results[results["Valid"] == True]
        y_true = results["density"].values
        y_pred = results["Predicted_Density"].values
    else:
        y_true = test_df["density"].values
        y_pred = np.array(predictions)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
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
    """Evaluate the model on training set."""
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TRAINING SET")
    print("="*70)
    
    predictions = predictor.predict(train_df["SMILES"].tolist())
    
    if len(predictions) != len(train_df):
        results = predictor.predict_with_details(train_df["SMILES"].tolist())
        results = results.merge(train_df[["SMILES", "density"]], on="SMILES", how="left")
        results = results[results["Valid"] == True]
        y_true = results["density"].values
        y_pred = results["Predicted_Density"].values
    else:
        y_true = train_df["density"].values
        y_pred = np.array(predictions)
    
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


def analyze_density_ranges(test_df, y_true, y_pred):
    """Analyze performance across different density ranges."""
    print("\n" + "="*70)
    print("PERFORMANCE BY DENSITY RANGE")
    print("="*70)
    
    # Define density ranges
    ranges = [
        (0, 600, "Very Low (<600)"),
        (600, 800, "Low (600-800)"),
        (800, 1000, "Medium (800-1000)"),
        (1000, 1200, "High (1000-1200)"),
        (1200, float('inf'), "Very High (>1200)")
    ]
    
    print(f"\n{'Range':<25} {'Count':>8} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    print("-" * 70)
    
    for low, high, label in ranges:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() == 0:
            continue
        
        y_true_range = y_true[mask]
        y_pred_range = y_pred[mask]
        
        rmse = np.sqrt(mean_squared_error(y_true_range, y_pred_range))
        mae = mean_absolute_error(y_true_range, y_pred_range)
        mape = np.mean(np.abs((y_true_range - y_pred_range) / y_true_range)) * 100
        
        print(f"{label:<25} {mask.sum():>8} {rmse:>10.2f} {mae:>10.2f} {mape:>9.2f}%")


def analyze_by_molecular_properties(test_df, y_true, y_pred):
    """Analyze performance by molecular properties."""
    print("\n" + "="*70)
    print("PERFORMANCE BY MOLECULAR PROPERTIES")
    print("="*70)
    
    # Calculate molecular properties
    mol_props = calculate_molecular_properties(test_df["SMILES"].tolist())
    
    # Add predictions and errors
    mol_props['Actual'] = y_true
    mol_props['Predicted'] = y_pred
    mol_props['Error'] = np.abs(y_true - y_pred)
    mol_props['Relative_Error'] = np.abs((y_true - y_pred) / y_true) * 100
    
    # Analyze by molecular weight
    print("\n--- By Molecular Weight ---")
    mw_ranges = [
        (0, 50, "Very Light (<50)"),
        (50, 100, "Light (50-100)"),
        (100, 200, "Medium (100-200)"),
        (200, 300, "Heavy (200-300)"),
        (300, float('inf'), "Very Heavy (>300)")
    ]
    
    print(f"{'Range':<25} {'Count':>8} {'Avg Error':>12} {'Avg MAPE':>12}")
    print("-" * 60)
    
    for low, high, label in mw_ranges:
        mask = (mol_props['MolWt'] >= low) & (mol_props['MolWt'] < high)
        if mask.sum() == 0:
            continue
        
        avg_error = mol_props.loc[mask, 'Error'].mean()
        avg_mape = mol_props.loc[mask, 'Relative_Error'].mean()
        
        print(f"{label:<25} {mask.sum():>8} {avg_error:>12.2f} {avg_mape:>11.2f}%")
    
    # Analyze by number of rings
    print("\n--- By Ring Count ---")
    print(f"{'Rings':<25} {'Count':>8} {'Avg Error':>12} {'Avg MAPE':>12}")
    print("-" * 60)
    
    for n_rings in sorted(mol_props['NumRings'].dropna().unique()):
        mask = mol_props['NumRings'] == n_rings
        if mask.sum() < 3:  # Skip if too few samples
            continue
        
        avg_error = mol_props.loc[mask, 'Error'].mean()
        avg_mape = mol_props.loc[mask, 'Relative_Error'].mean()
        
        label = f"{int(n_rings)} ring(s)"
        print(f"{label:<25} {mask.sum():>8} {avg_error:>12.2f} {avg_mape:>11.2f}%")
    
    return mol_props


def create_enhanced_plots(y_true, y_pred, mol_props, save_path="enhanced_evaluation.png"):
    """Create comprehensive visualization."""
    print("\n" + "="*70)
    print("CREATING ENHANCED PLOTS")
    print("="*70)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main scatter plot
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(y_true, y_pred, alpha=0.5, s=30, c=mol_props['MolWt'], cmap='viridis')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('Actual Density (kg/m³)', fontsize=11)
    ax1.set_ylabel('Predicted Density (kg/m³)', fontsize=11)
    ax1.set_title('Predicted vs Actual (colored by MW)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residual plot
    ax2 = fig.add_subplot(gs[0, 2])
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Density', fontsize=10)
    ax2.set_ylabel('Residuals', fontsize=10)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Residuals', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error by molecular weight
    ax4 = fig.add_subplot(gs[1, 1])
    errors = np.abs(y_true - y_pred)
    ax4.scatter(mol_props['MolWt'], errors, alpha=0.5, s=20, c='coral')
    ax4.set_xlabel('Molecular Weight (g/mol)', fontsize=10)
    ax4.set_ylabel('Absolute Error', fontsize=10)
    ax4.set_title('Error vs Molecular Weight', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Error by density range
    ax5 = fig.add_subplot(gs[1, 2])
    bins = [0, 600, 800, 1000, 1200, 1600]
    y_true_binned = pd.cut(y_true, bins=bins)
    error_by_bin = pd.DataFrame({'bin': y_true_binned, 'error': errors})
    error_by_bin.boxplot(column='error', by='bin', ax=ax5)
    ax5.set_xlabel('Density Range', fontsize=10)
    ax5.set_ylabel('Absolute Error', fontsize=10)
    ax5.set_title('Error Distribution by Density Range', fontsize=12, fontweight='bold')
    plt.sca(ax5)
    plt.xticks(rotation=45, ha='right')
    
    # 6. Relative error distribution
    ax6 = fig.add_subplot(gs[2, 0])
    relative_errors = np.abs((y_true - y_pred) / y_true) * 100
    ax6.hist(relative_errors, bins=40, edgecolor='black', alpha=0.7, color='green')
    ax6.axvline(x=np.median(relative_errors), color='r', linestyle='--', lw=2, label=f'Median: {np.median(relative_errors):.2f}%')
    ax6.set_xlabel('Relative Error (%)', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Relative Error Distribution', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Error by ring count
    ax7 = fig.add_subplot(gs[2, 1])
    ring_error = mol_props.groupby('NumRings')['Error'].mean()
    ax7.bar(ring_error.index, ring_error.values, color='purple', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Number of Rings', fontsize=10)
    ax7.set_ylabel('Average Absolute Error', fontsize=10)
    ax7.set_title('Error vs Ring Count', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Cumulative error distribution
    ax8 = fig.add_subplot(gs[2, 2])
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax8.plot(sorted_errors, cumulative, linewidth=2)
    ax8.axvline(x=np.percentile(errors, 90), color='r', linestyle='--', label=f'90th percentile: {np.percentile(errors, 90):.2f}')
    ax8.set_xlabel('Absolute Error', fontsize=10)
    ax8.set_ylabel('Cumulative %', fontsize=10)
    ax8.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Enhanced plots saved to {save_path}")
    
    return fig


def analyze_worst_predictions(test_df, y_true, y_pred, mol_props, top_n=15):
    """Deep dive into worst predictions."""
    print("\n" + "="*70)
    print("DETAILED ERROR ANALYSIS")
    print("="*70)
    
    errors = np.abs(y_true - y_pred)
    worst_indices = np.argsort(errors)[-top_n:][::-1]
    
    print(f"\nTop {top_n} worst predictions with molecular properties:")
    print("-" * 120)
    print(f"{'SMILES':<25} {'Actual':>8} {'Pred':>8} {'Error':>8} {'Error%':>8} {'MW':>8} {'Rings':>7} {'RotBonds':>9}")
    print("-" * 120)
    
    for idx in worst_indices:
        smiles = test_df.iloc[idx]["SMILES"]
        actual = y_true[idx]
        predicted = y_pred[idx]
        error = errors[idx]
        error_pct = error / actual * 100
        
        mw = mol_props.iloc[idx]['MolWt']
        rings = mol_props.iloc[idx]['NumRings']
        rot_bonds = mol_props.iloc[idx]['NumRotatableBonds']
        
        smiles_display = smiles[:22] + "..." if len(smiles) > 25 else smiles
        print(f"{smiles_display:<25} {actual:>8.2f} {predicted:>8.2f} {error:>8.2f} {error_pct:>7.1f}% {mw:>8.1f} {rings:>7.0f} {rot_bonds:>9.0f}")


def error_correlation_analysis(mol_props):
    """Analyze correlations between molecular properties and errors."""
    print("\n" + "="*70)
    print("ERROR CORRELATION WITH MOLECULAR PROPERTIES")
    print("="*70)
    
    properties = ['MolWt', 'NumHeavyAtoms', 'NumRotatableBonds', 'NumRings', 'NumAromaticRings', 'TPSA', 'logP']
    
    print(f"\n{'Property':<25} {'Correlation with Error':>25}")
    print("-" * 50)
    
    for prop in properties:
        if prop in mol_props.columns:
            corr = mol_props[['Error', prop]].dropna().corr().iloc[0, 1]
            print(f"{prop:<25} {corr:>25.4f}")


def main():
    """Main enhanced evaluation pipeline."""
    print("="*70)
    print("ENHANCED DENSITY MODEL EVALUATION")
    print("="*70)
    
    predictor = DensityPredictor()
    train_df, test_df = load_and_split_data(test_size=0.2, random_state=42)
    
    # Basic evaluation
    train_metrics = evaluate_training_set(predictor, train_df)
    y_true, y_pred, test_metrics = evaluate_model(predictor, test_df)
    
    # Overfitting check
    print("\n" + "="*70)
    print("OVERFITTING CHECK")
    print("="*70)
    print(f"Train RMSE: {train_metrics['RMSE']:.4f}")
    print(f"Test RMSE:  {test_metrics['RMSE']:.4f}")
    print(f"Difference: {test_metrics['RMSE'] - train_metrics['RMSE']:.4f}")
    
    if test_metrics['RMSE'] > train_metrics['RMSE'] * 1.2:
        print("⚠ Warning: Model may be overfitting")
    else:
        print("✓ Model generalization looks good!")
    
    # Advanced analysis
    analyze_density_ranges(test_df, y_true, y_pred)
    mol_props = analyze_by_molecular_properties(test_df, y_true, y_pred)
    analyze_worst_predictions(test_df, y_true, y_pred, mol_props, top_n=15)
    error_correlation_analysis(mol_props)
    
    # Enhanced visualizations
    create_enhanced_plots(y_true, y_pred, mol_props, 
                         save_path=os.path.join(PROJECT_ROOT, "density_model/enhanced_evaluation.png"))
    
    print("\n" + "="*70)
    print("ENHANCED EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()