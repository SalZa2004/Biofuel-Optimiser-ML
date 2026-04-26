"""
mixture_kl_analysis.py

Converts both the mixture training dataset and FAME biodiesel dataset into
distributions over a shared (carbon_count, unsaturation) feature space, then
uses symmetric KL divergence to find the closest FAME match for every mixture
and compares their CN values.

Feature space
-------------
Each molecule is mapped to a bin (carbon_count, unsaturation_level) where:
  - carbon_count  : number of carbon atoms in the molecule
  - unsaturation  : C=C double bonds + ring count, capped at 3
    (0 = fully saturated, 1 = mono, 2 = di, 3+ = poly/aromatic)

FAME rows already encode this via 'C18:1' notation.
Mixture rows are represented as mole-fraction-weighted sums over their components.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Data paths ────────────────────────────────────────────────────────────────
MIX_PATH  = "data/database/mixture_training_dataset.csv"
FAME_PATH = "data/database/fame_cleaned_normalised_2.csv"
OUT_CSV   = "results/kl_cn_comparison.csv"
OUT_PLOT  = "results/kl_cn_comparison.png"

# ── FAME column definitions ───────────────────────────────────────────────────
FAME_COLS = [
    "C8:0","C10:0","C12:0","C14:0","C16:0","C16:1","C17:0",
    "C18:0","C18:1","C18:2","C18:3","C20:0","C20:1",
    "C22:0","C22:1","C24:0","C24:1",
]

def parse_fame_key(col: str) -> tuple[int, int]:
    """'C18:1' → (18, 1)"""
    c, u = col[1:].split(":")
    return (int(c), int(u))

FAME_BIN_KEYS = [parse_fame_key(c) for c in FAME_COLS]

# ── Molecule → (carbon_count, unsaturation) ───────────────────────────────────
def inchi_to_bin(inchi: str) -> tuple[int, int] | None:
    mol = Chem.inchi.MolFromInchi(inchi.strip())
    if mol is None:
        return None
    c_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    cc_double = sum(
        1 for b in mol.GetBonds()
        if b.GetBondTypeAsDouble() == 2.0
        and b.GetBeginAtom().GetAtomicNum() == 6
        and b.GetEndAtom().GetAtomicNum() == 6
    )
    rings = mol.GetRingInfo().NumRings()
    unsat = min(cc_double + rings, 3)
    return (c_count, unsat)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading datasets…")
mix_df  = pd.read_csv(MIX_PATH,  encoding="latin1")
fame_df = pd.read_csv(FAME_PATH, encoding="utf-8-sig")

print(f"  Mixture rows (raw) : {len(mix_df)}")
# Drop rows with no fuel components or no experimental CN
mix_df = mix_df.dropna(subset=["fuel1 inchi", "Experimental value (from literature)"]).reset_index(drop=True)
print(f"  Mixture rows (valid CN + components): {len(mix_df)}")
print(f"  FAME rows    : {len(fame_df)}")

# ── Cache component bins ──────────────────────────────────────────────────────
all_inchis: set[str] = set()
for col in ["fuel1 inchi", "fuel2 inchi", "fuel3 inchi", "fuel4 inchi"]:
    all_inchis.update(mix_df[col].dropna().str.strip().unique())

component_bin: dict[str, tuple[int, int]] = {}
failed = []
for inchi in all_inchis:
    b = inchi_to_bin(inchi)
    if b:
        component_bin[inchi] = b
    else:
        failed.append(inchi[:60])

if failed:
    print(f"  ⚠  Could not parse {len(failed)} InChI(s): {failed}")

# ── Build unified bin index ───────────────────────────────────────────────────
# Union of FAME bins and mixture component bins
mix_bins = set(component_bin.values())
all_bins = sorted(set(FAME_BIN_KEYS) | mix_bins)   # sorted by (carbon, unsat)
bin_index = {b: i for i, b in enumerate(all_bins)}
n_bins = len(all_bins)
print(f"\n  Shared feature space: {n_bins} bins")
print("  Bins:", [f"C{c}:{u}" for c, u in all_bins])

# ── Mixture row → distribution vector ────────────────────────────────────────
def mixture_to_vector(row: pd.Series) -> np.ndarray:
    """
    Builds a normalised distribution over bin_index for one mixture row.
    Missing molar fractions for the last component(s) are inferred as the
    remainder of 1 − sum(known fractions).
    """
    components = []
    for i in range(1, 5):
        inchi = row.get(f"fuel{i} inchi")
        if pd.isna(inchi):
            continue
        inchi = str(inchi).strip()
        frac_raw = row.get(f"molar fraction fuel {i}", np.nan)
        frac = float(frac_raw) if not pd.isna(frac_raw) else None
        components.append((inchi, frac))

    # Infer missing fraction (binary blend: only frac1 stored → frac2 = 1-frac1)
    known_sum = sum(f for _, f in components if f is not None)
    n_unknown  = sum(1 for _, f in components if f is None)
    remainder  = max(0.0, 1.0 - known_sum)
    per_unknown = (remainder / n_unknown) if n_unknown else 0.0
    components = [(inchi, f if f is not None else per_unknown)
                  for inchi, f in components]

    vec = np.zeros(n_bins)
    for inchi, frac in components:
        b = component_bin.get(inchi)
        if b is not None and frac > 0:
            vec[bin_index[b]] += frac

    total = vec.sum()
    return vec / total if total > 0 else vec

# ── FAME row → distribution vector ───────────────────────────────────────────
def fame_to_vector(row: pd.Series) -> np.ndarray:
    vec = np.zeros(n_bins)
    for col, key in zip(FAME_COLS, FAME_BIN_KEYS):
        val = row.get(col, 0)
        if not pd.isna(val) and val > 0:
            vec[bin_index[key]] += float(val)
    total = vec.sum()
    return vec / total if total > 0 else vec

# ── Build vector matrices ─────────────────────────────────────────────────────
print("\nBuilding distribution vectors…")
mix_vecs  = np.array([mixture_to_vector(row) for _, row in mix_df.iterrows()])
fame_vecs = np.array([fame_to_vector(row)    for _, row in fame_df.iterrows()])

# ── Symmetric KL divergence ───────────────────────────────────────────────────
EPSILON = 1e-10

def sym_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric KL divergence: (KL(p‖q) + KL(q‖p)) / 2"""
    p = (p + EPSILON); p /= p.sum()
    q = (q + EPSILON); q /= q.sum()
    return 0.5 * (float(np.sum(rel_entr(p, q))) + float(np.sum(rel_entr(q, p))))

print("Computing pairwise KL divergences…")
kl_matrix = np.zeros((len(mix_vecs), len(fame_vecs)))
for i, pv in enumerate(mix_vecs):
    for j, qv in enumerate(fame_vecs):
        kl_matrix[i, j] = sym_kl(pv, qv)

# ── For each FAME, find its closest mixture ───────────────────────────────────
# kl_matrix shape: (n_mixtures, n_fames) → argmin over axis=0 gives best mixture per FAME
best_mix_idx = kl_matrix.argmin(axis=0)   # shape: (n_fames,)
best_kl      = kl_matrix.min(axis=0)      # shape: (n_fames,)

mix_cn  = mix_df["Experimental value (from literature)"].values
fame_cn = fame_df["CN"].values

results = pd.DataFrame({
    "biodiesel"       : fame_df["Biodiesel"].values,
    "fame_cn"         : fame_cn,
    "best_mix_index"  : best_mix_idx,
    "best_mix_cn"     : mix_cn[best_mix_idx],
    "kl_divergence"   : best_kl,
    "cn_difference"   : fame_cn - mix_cn[best_mix_idx],
})

results.to_csv(OUT_CSV, index=False)
print(f"\nSaved results → {OUT_CSV}")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n── CN Comparison Summary (FAME → best mixture) ──────────────────────")
print(f"  FAMEs matched    : {len(results)}")
print(f"  Mean |ΔCN|       : {results['cn_difference'].abs().mean():.2f}")
print(f"  Median |ΔCN|     : {results['cn_difference'].abs().median():.2f}")
print(f"  Mean KL div      : {results['kl_divergence'].mean():.4f}")
print(f"\n  All matches sorted by KL divergence (closest first):")
print(results.sort_values("kl_divergence")[
    ["biodiesel","fame_cn","best_mix_cn","kl_divergence","cn_difference"]
].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: scatter FAME CN vs matched mixture CN, coloured by KL
sc = axes[0].scatter(
    results["best_mix_cn"], results["fame_cn"],
    c=results["kl_divergence"], cmap="viridis_r",
    alpha=0.8, edgecolors="k", linewidths=0.3, s=40,
)
plt.colorbar(sc, ax=axes[0], label="Symmetric KL divergence")
lims = [min(results["best_mix_cn"].min(), results["fame_cn"].min()) - 2,
        max(results["best_mix_cn"].max(), results["fame_cn"].max()) + 2]
axes[0].plot(lims, lims, "r--", lw=1, label="y = x (perfect agreement)")
axes[0].set_xlim(lims); axes[0].set_ylim(lims)
axes[0].set_xlabel("Best matching mixture CN (experimental)")
axes[0].set_ylabel("Biodiesel FAME CN")
axes[0].set_title("FAME CN vs Closest Mixture CN\n(by KL divergence in C/unsaturation space)")
axes[0].legend()

# Right: distribution of ΔCN, separated into low/high KL groups
kl_thresh = np.percentile(results["kl_divergence"], 25)
close  = results[results["kl_divergence"] <= kl_thresh]["cn_difference"]
far    = results[results["kl_divergence"] >  kl_thresh]["cn_difference"]
bins   = np.linspace(results["cn_difference"].min() - 1,
                     results["cn_difference"].max() + 1, 40)
axes[1].hist(far,   bins=bins, alpha=0.5, color="steelblue", label=f"High KL (top 75%)")
axes[1].hist(close, bins=bins, alpha=0.7, color="darkorange", label=f"Low KL (bottom 25%)")
axes[1].axvline(0, color="red", lw=1, linestyle="--")
axes[1].set_xlabel("ΔCN  (FAME CN − matched mixture CN)")
axes[1].set_ylabel("Count")
axes[1].set_title("CN Difference Distribution\n(low KL = structurally closer match)")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150)
print(f"\nSaved plot → {OUT_PLOT}")
plt.show()
