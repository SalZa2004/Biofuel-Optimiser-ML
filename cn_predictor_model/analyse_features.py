"""
Analyze which features were selected by the FeatureSelector.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train import FeatureSelector, DESCRIPTOR_NAMES
import joblib
import os


def analyze_selected_features(selector_path='cn_model/artifacts/selector.joblib'):
    """
    Analyze and display the features selected by the FeatureSelector.
    """
    
    print("="*70)
    print("FEATURE SELECTION ANALYSIS")
    print("="*70)
    
    # Load the selector
    if not os.path.exists(selector_path):
        print(f"\n❌ Error: Selector not found at {selector_path}")
        print("Please train the model first: python train.py train")
        return
    
    selector = FeatureSelector.load(selector_path)
    
    # Basic info
    print(f"\nTotal Morgan fingerprint bits: {selector.n_morgan}")
    print(f"Correlation threshold: {selector.corr_threshold}")
    print(f"Top K features selected: {selector.top_k}")
    
    # Feature breakdown
    print("\n" + "="*70)
    print("FEATURE SELECTION PIPELINE")
    print("="*70)
    
    # Step 1: Original features
    n_original_descriptors = len(DESCRIPTOR_NAMES)
    n_original_total = selector.n_morgan + n_original_descriptors
    
    print(f"\n1. Original Features:")
    print(f"   - Morgan fingerprint bits: {selector.n_morgan}")
    print(f"   - RDKit descriptors: {n_original_descriptors}")
    print(f"   - Total original features: {n_original_total}")
    
    # Step 2: After correlation filtering
    n_removed_corr = len(selector.corr_cols_to_drop)
    n_after_corr = selector.n_morgan + (n_original_descriptors - n_removed_corr)
    
    print(f"\n2. After Correlation Filtering (threshold={selector.corr_threshold}):")
    print(f"   - Removed correlated descriptors: {n_removed_corr}")
    print(f"   - Remaining features: {n_after_corr}")
    
    # Step 3: After importance-based selection
    n_selected = len(selector.selected_indices)
    
    print(f"\n3. After Importance-Based Selection:")
    print(f"   - Selected top features: {n_selected}")
    print(f"   - Reduction: {n_original_total} → {n_selected} ({n_selected/n_original_total*100:.1f}% retained)")
    
    # Analyze which features were selected
    print("\n" + "="*70)
    print("SELECTED FEATURE BREAKDOWN")
    print("="*70)
    
    # Determine how many Morgan bits vs descriptors were selected
    n_morgan_selected = sum(1 for idx in selector.selected_indices if idx < selector.n_morgan)
    n_desc_selected = n_selected - n_morgan_selected
    
    print(f"\nSelected features by type:")
    print(f"   - Morgan fingerprint bits: {n_morgan_selected} / {selector.n_morgan} ({n_morgan_selected/selector.n_morgan*100:.1f}%)")
    print(f"   - RDKit descriptors: {n_desc_selected} / {n_original_descriptors - n_removed_corr} ({n_desc_selected/(n_original_descriptors - n_removed_corr)*100:.1f}%)")
    
    # Get descriptor names that weren't removed by correlation
    remaining_desc_names = [name for i, name in enumerate(DESCRIPTOR_NAMES) 
                           if i not in selector.corr_cols_to_drop]
    
    # Find which descriptors were selected
    selected_descriptors = []
    for idx in selector.selected_indices:
        if idx >= selector.n_morgan:
            desc_idx = idx - selector.n_morgan
            if desc_idx < len(remaining_desc_names):
                selected_descriptors.append(remaining_desc_names[desc_idx])
    
    print(f"\n" + "="*70)
    print(f"SELECTED RDKIT DESCRIPTORS ({len(selected_descriptors)} total)")
    print("="*70)
    
    if len(selected_descriptors) > 0:
        print("\nDescriptor names:")
        for i, desc in enumerate(selected_descriptors, 1):
            print(f"  {i:3d}. {desc}")
    else:
        print("\nNo RDKit descriptors were selected (only Morgan fingerprint bits)")
    
    # Save to file
    results = {
        "Total_Original_Features": n_original_total,
        "Morgan_Bits_Original": selector.n_morgan,
        "Descriptors_Original": n_original_descriptors,
        "Correlated_Removed": n_removed_corr,
        "Features_After_Correlation": n_after_corr,
        "Final_Selected_Features": n_selected,
        "Morgan_Bits_Selected": n_morgan_selected,
        "Descriptors_Selected": n_desc_selected,
        "Selected_Descriptor_Names": selected_descriptors
    }
    
    # Save summary
    summary_path = "cn_model/feature_selection_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FEATURE SELECTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("Pipeline:\n")
        f.write(f"  Original features: {n_original_total}\n")
        f.write(f"  After correlation filter: {n_after_corr}\n")
        f.write(f"  Final selected: {n_selected}\n\n")
        
        f.write("Selected features by type:\n")
        f.write(f"  Morgan fingerprint bits: {n_morgan_selected}\n")
        f.write(f"  RDKit descriptors: {n_desc_selected}\n\n")
        
        f.write("Selected RDKit Descriptors:\n")
        for i, desc in enumerate(selected_descriptors, 1):
            f.write(f"  {i:3d}. {desc}\n")
    
    print(f"\n✓ Summary saved to {summary_path}")
    
    # Save selected descriptors to CSV
    if len(selected_descriptors) > 0:
        desc_df = pd.DataFrame({"Descriptor_Name": selected_descriptors})
        desc_csv_path = "cn_model/selected_descriptors.csv"
        desc_df.to_csv(desc_csv_path, index=False)
        print(f"✓ Selected descriptors saved to {desc_csv_path}")
    
    return results


def plot_feature_distribution(selector_path='cn_model/artifacts/selector.joblib'):
    """
    Create visualization of feature selection.
    """
    
    if not os.path.exists(selector_path):
        return
    
    selector = FeatureSelector.load(selector_path)
    
    # Count Morgan bits vs descriptors in selected features
    n_morgan_selected = sum(1 for idx in selector.selected_indices if idx < selector.n_morgan)
    n_desc_selected = len(selector.selected_indices) - n_morgan_selected
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Pie chart of selected features
    ax1 = axes[0]
    labels = ['Morgan Bits', 'RDKit Descriptors']
    sizes = [n_morgan_selected, n_desc_selected]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Selected Features by Type', fontsize=14, fontweight='bold')
    
    # 2. Bar chart showing reduction
    ax2 = axes[1]
    
    n_original_descriptors = len(DESCRIPTOR_NAMES)
    n_original_total = selector.n_morgan + n_original_descriptors
    n_removed_corr = len(selector.corr_cols_to_drop)
    n_after_corr = selector.n_morgan + (n_original_descriptors - n_removed_corr)
    n_selected = len(selector.selected_indices)
    
    stages = ['Original', 'After\nCorrelation\nFilter', 'Final\nSelected']
    counts = [n_original_total, n_after_corr, n_selected]
    colors_bar = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax2.bar(stages, counts, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Features', fontsize=12)
    ax2.set_title('Feature Reduction Pipeline', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = "cn_model/feature_selection_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {plot_path}")
    
    return fig


def get_feature_importance_from_model(model_path='cn_model/artifacts/model.joblib',
                                     selector_path='cn_model/artifacts/selector.joblib'):
    """
    Get feature importances from the trained model.
    """
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    if not os.path.exists(model_path) or not os.path.exists(selector_path):
        print("\n❌ Error: Model or selector not found")
        return
    
    # Load model and selector
    model = joblib.load(model_path)
    selector = FeatureSelector.load(selector_path)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Get descriptor names
    remaining_desc_names = [name for i, name in enumerate(DESCRIPTOR_NAMES) 
                           if i not in selector.corr_cols_to_drop]
    
    # Map importances to feature names
    feature_info = []
    for i, (idx, imp) in enumerate(zip(selector.selected_indices, importances)):
        if idx < selector.n_morgan:
            feature_name = f"Morgan_Bit_{idx}"
            feature_type = "Morgan"
        else:
            desc_idx = idx - selector.n_morgan
            if desc_idx < len(remaining_desc_names):
                feature_name = remaining_desc_names[desc_idx]
                feature_type = "Descriptor"
            else:
                feature_name = f"Unknown_{idx}"
                feature_type = "Unknown"
        
        feature_info.append({
            "Feature_Name": feature_name,
            "Feature_Type": feature_type,
            "Importance": imp,
            "Original_Index": idx
        })
    
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame(feature_info)
    importance_df = importance_df.sort_values("Importance", ascending=False)
    
    print(f"\nTop 20 Most Important Features:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature Name':<40} {'Type':<12} {'Importance':<10}")
    print("-" * 80)
    
    for i, row in importance_df.head(20).iterrows():
        print(f"{importance_df.index.get_loc(i)+1:<6} {row['Feature_Name'][:40]:<40} "
              f"{row['Feature_Type']:<12} {row['Importance']:.6f}")
    
    # Save full importance list
    importance_path = "cn_model/feature_importances.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\n✓ Full feature importance list saved to {importance_path}")
    
    # Plot top features
    plot_top_features(importance_df)
    
    return importance_df


def plot_top_features(importance_df, top_n=20):
    """
    Plot the most important features.
    """
    
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by type
    colors = ['#ff6b6b' if t == 'Morgan' else '#4ecdc4' 
              for t in top_features['Feature_Type']]
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['Importance'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Truncate long names
    labels = [name[:35] + '...' if len(name) > 35 else name 
              for name in top_features['Feature_Name']]
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', label='Morgan Fingerprint'),
        Patch(facecolor='#4ecdc4', label='RDKit Descriptor')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    plot_path = "cn_model/top_features.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Top features plot saved to {plot_path}")


def main():
    """
    Main analysis pipeline.
    """
    
    print("="*70)
    print("FEATURE SELECTION ANALYSIS")
    print("="*70)
    
    # Analyze feature selection
    results = analyze_selected_features()
    
    if results:
        # Create visualizations
        plot_feature_distribution()
        
        # Analyze feature importances from model
        get_feature_importance_from_model()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  - cn_model/feature_selection_summary.txt")
        print("  - cn_model/selected_descriptors.csv")
        print("  - cn_model/feature_selection_visualization.png")
        print("  - cn_model/feature_importances.csv")
        print("  - cn_model/top_features.png")
        print("="*70)


if __name__ == "__main__":
    main()