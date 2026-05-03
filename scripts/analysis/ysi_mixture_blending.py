import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
# =============================================================
# Das et al. 2017 Table 1 - YSI values (hexane=0, benzene=100)
# =============================================================
das_table1 = {
    'benzene':                          100.0,
    'n-hexane':                         0.0,
    'toluene':                          172.5,
    'methylcyclohexane':                20.8,
    'm-xylene':                         193.1,
    '1,3-dimethylbenzene':              193.1,
    'ethylbenzene':                     203.4,
    '2,2,4-trimethylpentane':           23.8,
    'isooctane':                        23.8,
    'iso-octane':                       23.8,
    'propylbenzene':                    181.0,
    'n-propylbenzene':                  181.0,
    '1,2,4-trimethylbenzene':           260.6,
    '1,3,5-trimethylbenzene':           243.3,
    'mesitylene':                       243.3,
    '1,2-dihydronaphthalene':           352.1,
    '1,2,3,4-tetrahydronaphthalene':    264.4,
    'tetralin':                         264.4,
    'decahydronaphthalene':             53.7,
    'decalin':                          53.7,
    'n-butylcyclohexane':               17.0,
    'n-decane':                         7.4,
    '1-methylnaphthalene':              471.2,
    'cyclohexylbenzene':                224.4,
    'n-dodecane':                       9.8,
    'n-tetradecane':                    9.5,
    'n-hexadecane':                     11.7,
    'iso-cetane':                       31.0,
    '2,2,4,4,6,8,8-heptamethylnonane':  31.0,
    'n-octadecane':                     15.3,
    'n-eicosane':                       14.1,
}

# =============================================================
# Load unified YSI database
# =============================================================
db_path = '/home/salvina2004/biofuel-ml/data/database/merged_CN_YSI_SMILES_Cleaned.csv'
df = pd.read_csv(db_path)
df_ysi = df[df['YSI_Unified_Measured'].notna()].copy()
df_ysi['Name_lower'] = df_ysi['Name'].str.lower().str.strip()
print(f"Unified DB: {len(df_ysi)} compounds with measured YSI")

# =============================================================
# Match compounds - direct name match first, then partial
# =============================================================
matches = []
matched_unified_names = set()

for das_name, das_ysi_val in das_table1.items():
    match = df_ysi[df_ysi['Name_lower'] == das_name.lower()]
    if len(match) > 0:
        row = match.iloc[0]
        if row['Name'] not in matched_unified_names:
            matches.append({
                'das_name': das_name,
                'unified_name': row['Name'],
                'das_ysi': das_ysi_val,
                'unified_ysi': row['YSI_Unified_Measured'],
                'match_type': 'exact'
            })
            matched_unified_names.add(row['Name'])

# Partial matching for remaining
for das_name, das_ysi_val in das_table1.items():
    if any(m['das_name'] == das_name for m in matches):
        continue
    for _, row in df_ysi.iterrows():
        if row['Name'] in matched_unified_names:
            continue
        if das_name.lower() in row['Name_lower'] or row['Name_lower'] in das_name.lower():
            matches.append({
                'das_name': das_name,
                'unified_name': row['Name'],
                'das_ysi': das_ysi_val,
                'unified_ysi': row['YSI_Unified_Measured'],
                'match_type': 'partial'
            })
            matched_unified_names.add(row['Name'])
            break

matches_df = pd.DataFrame(matches)
print(f"\nFound {len(matches_df)} matching compounds:")
print(matches_df[['das_name', 'unified_name', 'das_ysi', 'unified_ysi', 'match_type']].to_string())

# =============================================================
# Fit linear rescaling
# =============================================================
if len(matches_df) < 2:
    print("\nNot enough matches — check name matching above")
else:
    slope, intercept, r, p, se = stats.linregress(
        matches_df['das_ysi'],
        matches_df['unified_ysi']
    )
    r2 = r**2
    print(f"\n=== Rescaling fit ===")
    print(f"slope     = {slope:.4f}")
    print(f"intercept = {intercept:.4f}")
    print(f"R²        = {r2:.4f}")
    print(f"\nYSI_unified = {slope:.4f} * YSI_das + ({intercept:.4f})")

    # Check if scales are actually different
    if abs(slope - 1.0) < 0.05 and abs(intercept) < 5:
        print("\nScales appear very similar — rescaling may not be the main source of error")
    else:
        print(f"\nScales differ significantly — rescaling is likely important")

    # =============================================================
    # Plot
    # =============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    colors = ['steelblue' if m == 'exact' else 'orange' for m in matches_df['match_type']]
    ax.scatter(matches_df['das_ysi'], matches_df['unified_ysi'],
               c=colors, s=70, zorder=3)
    x_line = np.linspace(0, matches_df['das_ysi'].max() * 1.05, 100)
    ax.plot(x_line, slope * x_line + intercept, 'r--',
            label=f'y = {slope:.3f}x + {intercept:.2f}\nR² = {r2:.3f}')
    ax.plot(x_line, x_line, 'k:', alpha=0.3, label='1:1 line')
    for _, row in matches_df.iterrows():
        ax.annotate(row['das_name'], (row['das_ysi'], row['unified_ysi']),
                    fontsize=7, ha='left', va='bottom', alpha=0.7)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='steelblue', label='Exact match'),
        Patch(color='orange', label='Partial match'),
        plt.Line2D([0], [0], color='r', linestyle='--', label=f'Fit: R²={r2:.3f}'),
        plt.Line2D([0], [0], color='k', linestyle=':', alpha=0.3, label='1:1'),
    ])
    ax.set_xlabel('YSI (Das et al. 2017 scale)')
    ax.set_ylabel('YSI (Unified scale)')
    ax.set_title('YSI Scale Comparison')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    predicted = slope * matches_df['das_ysi'] + intercept
    residuals = matches_df['unified_ysi'] - predicted
    bar_colors = ['steelblue' if m == 'exact' else 'orange' for m in matches_df['match_type']]
    ax.bar(range(len(matches_df)), residuals, color=bar_colors, alpha=0.8)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xticks(range(len(matches_df)))
    ax.set_xticklabels(matches_df['das_name'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Residual (Unified - Predicted)')
    ax.set_title('Rescaling Residuals')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ysi_rescaling_fit.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved.")

    # =============================================================
    # Print function to use in pipeline
    # =============================================================
    print(f"""
=== Use this in your pipeline ===

def rescale_das_to_unified(ysi_das):
    return {slope:.4f} * ysi_das + ({intercept:.4f})

# Eq 7a with rescaling:
def predict_mixture_ysi_7a(components, rescale=True):
    \"\"\"components: list of (mass_fraction, das_ysi_value) tuples\"\"\"
    ysi_das = sum(w * y for w, y in components)
    if rescale:
        return rescale_das_to_unified(ysi_das)
    return ysi_das

# Note: since rescaling is linear, you can apply it before or after 
# summing - the result is identical. Applying after is cleaner.
""")