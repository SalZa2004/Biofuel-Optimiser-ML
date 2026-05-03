"""
Evaluate mixture YSI predictions against YSI_Unified_Measured and
YSI_Unified_Regression values from mixture_database.csv.

Two methods are compared:
  Method A — Pure predictor  : pure-component YSI from ML model, then linear blend
  Method B — Carbon types   : pure-component YSI from Eq 7b (Σ N_j × C_j), then Eq 7a

Both use mass-fraction-weighted blending (W_i) as per the paper.

Note: rows where fractions of SMILES-covered components do not sum to 1.0
are skipped by default (missing components in the DB).  Pass
--include-incomplete to normalise and include them (marked on the plot).

Usage (from project root):
    python scripts/evaluate_mixture_ysi.py
    python scripts/evaluate_mixture_ysi.py --include-incomplete

Output:
    results/mixture_ysi_actual_vs_pred.png
"""

import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.predictors.pure_component.generic import GenericPredictor
from core.predictors.pure_component.hf_models import load_models
from core.shared_features import featurize_df
from core.blending.blending_law import (
    blend_ysi_carbon_type,
    ysi_from_carbon_types,
    rescale_das_to_unified,
    YSI_CARBON_CONTRIBUTIONS,
)

DATA_PATH      = ROOT / "data" / "database" / "mixture_database.csv"
RESULTS_DIR    = ROOT / "results"
MAX_COMPONENTS = 11
FRAC_TOL       = 0.02

def load_ysi_mixtures(path):
    mixtures = []
    with open(path, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            reg  = row.get("YSI_Unified_Regression", "").strip()
            meas = row.get("YSI_Unified_Measured",   "").strip()
            if not reg and not meas:
                continue

            smiles, fracs, frac_total = [], [], 0.0
            for i in range(1, MAX_COMPONENTS + 1):
                frac_str = row.get(f"fraction_fuel_{i}", "").strip()
                smi      = row.get(f"fuel_{i}_smiles",   "").strip()
                if frac_str:
                    frac_total += float(frac_str)
                    if smi:
                        smiles.append(smi)
                        fracs.append(float(frac_str))

            if not smiles:
                continue

            mixtures.append({
                "name":       row.get("Name", ""),
                "smiles":     smiles,
                "fracs":      fracs,
                "frac_total": frac_total,
                "complete":   abs(sum(fracs) - 1.0) <= FRAC_TOL,
                "ysi_reg":    float(reg)  if reg  else None,
                "ysi_meas":   float(meas) if meas else None,
            })
    return mixtures


def predict_gnn(smiles, fracs, ysi_predictor):
    """Method A: Pure-component YSI + linear mass-fraction blend."""
    X = featurize_df(smiles, return_df=False)
    if X is None:
        return None
    vals = ysi_predictor.predict_from_features(X)
    if any(v is None for v in vals):
        return None
    return sum(f * y for f, y in zip(fracs, vals)), vals


def predict_carbon_type(smiles, fracs):
    """Method B: carbon-type YSI (Eq 7b) + linear mass-fraction blend (Eq 7a),
    with rescaling from Das scale to unified YSI scale."""
    pure_ysi = [ysi_from_carbon_types(smi) for smi in smiles]
    if any(v is None for v in pure_ysi):
        return None, pure_ysi
    blended = sum(f * y for f, y in zip(fracs, pure_ysi))
    return rescale_das_to_unified(blended), pure_ysi


def metrics_str(actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted) if len(actual) > 1 else float("nan")
    return mae, rmse, r2


def make_panel(ax, actual, predicted, complete_flags, method_label, ref_label):
    complete   = [(a, p) for a, p, c in zip(actual, predicted, complete_flags) if c]
    incomplete = [(a, p) for a, p, c in zip(actual, predicted, complete_flags) if not c]

    if complete:
        ac, pc = zip(*complete)
        ax.scatter(ac, pc, s=65, alpha=0.85, edgecolors="k", linewidths=0.5,
                   color="steelblue", zorder=3, label=f"Complete (n={len(complete)})")
    if incomplete:
        ai, pi = zip(*incomplete)
        ax.scatter(ai, pi, s=65, alpha=0.7, edgecolors="k", linewidths=0.5,
                   color="orange", marker="^", zorder=3,
                   label=f"Normalised (n={len(incomplete)})")

    lo = min(min(actual), min(predicted)) * 0.88
    hi = max(max(actual), max(predicted)) * 1.08
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="Ideal", zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    mae, rmse, r2 = metrics_str(actual, predicted)
    lines = [f"All  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}"]
    if complete and incomplete:
        ac2, pc2 = zip(*complete)
        mae_c, rmse_c, r2_c = metrics_str(list(ac2), list(pc2))
        lines.append(f"Complete  MAE={mae_c:.1f}  RMSE={rmse_c:.1f}  R²={r2_c:.3f}")

    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
            va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.set_xlabel(f"YSI ({ref_label})", fontsize=11)
    ax.set_ylabel("Predicted mixture YSI", fontsize=11)
    ax.set_title(method_label, fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")


def print_per_row(results):
    header = (f"{'Mixture':<45} {'Meas':>7} {'Reg':>7} "
              f"{'GNN':>7} {'Err_A':>7}  {'CT':>7} {'Err_B':>7}  {'Ens':>7} {'Err_C':>7}  Status")
    print(header)
    print("-" * len(header))
    for r in results:
        ref   = r["ysi_meas"] or r["ysi_reg"]
        err_a = f"{r['pred_gnn'] - ref:+.1f}" if r["pred_gnn"] is not None else "   fail"
        err_b = f"{r['pred_ct']  - ref:+.1f}" if r["pred_ct"]  is not None else "   fail"
        err_c = f"{r['pred_ens'] - ref:+.1f}" if r["pred_ens"] is not None else "   fail"
        meas  = f"{r['ysi_meas']:7.1f}" if r["ysi_meas"] else "      -"
        reg   = f"{r['ysi_reg']:7.1f}"  if r["ysi_reg"]  else "      -"
        gnn   = f"{r['pred_gnn']:7.1f}" if r["pred_gnn"] is not None else "   fail"
        ct    = f"{r['pred_ct']:7.1f}"  if r["pred_ct"]  is not None else "   fail"
        ens   = f"{r['pred_ens']:7.1f}" if r["pred_ens"] is not None else "   fail"
        flag  = "OK" if r["complete"] else f"partial({r['frac_total']:.2f})"
        print(f"{r['name'][:45]:<45} {meas} {reg} {gnn} {err_a:>7}  {ct} {err_b:>7}  {ens} {err_c:>7}  {flag}")

        for smi, frac, ysi_g, ysi_c in zip(
            r["smiles"], r["fracs"], r["pure_gnn"], r["pure_ct"]
        ):
            g = f"{ysi_g:.1f}" if ysi_g is not None else "fail"
            c = f"{ysi_c:.1f}" if ysi_c is not None else "fail"
            print(f"    {smi:<40} frac={frac:.3f}  GNN={g:>7}  CT={c:>7}")
        print()


def main():
    include_incomplete = "--include-incomplete" in sys.argv
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading {DATA_PATH.relative_to(ROOT)} ...")
    all_mixtures = load_ysi_mixtures(DATA_PATH)
    n_ok   = sum(1 for m in all_mixtures if m["complete"])
    n_part = sum(1 for m in all_mixtures if not m["complete"])
    print(f"  {len(all_mixtures)} rows with YSI data: {n_ok} complete, {n_part} with missing SMILES")

    if n_part:
        print("  Incomplete rows:")
        for m in all_mixtures:
            if not m["complete"]:
                print(f"    {m['name'][:65]}  (covered={sum(m['fracs']):.3f})")

    if include_incomplete:
        for m in all_mixtures:
            if not m["complete"]:
                total = sum(m["fracs"])
                m["fracs"] = [f / total for f in m["fracs"]]
        mixtures = all_mixtures
        print("\n  --include-incomplete: fractions normalised for partial rows")
    else:
        mixtures = [m for m in all_mixtures if m["complete"]]
        print(f"\n  Evaluating {len(mixtures)} complete rows "
              f"(pass --include-incomplete for all)")

    if not mixtures:
        print("Nothing to evaluate."); return

    print("\nLoading GNN YSI predictor ...")
    paths = load_models()
    ysi_predictor = GenericPredictor(paths["ysi"], "YSI")

    print("Predicting ...\n")
    results = []
    for mix in mixtures:
        gnn_result = predict_gnn(mix["smiles"], mix["fracs"], ysi_predictor)
        ct_result  = predict_carbon_type(mix["smiles"], mix["fracs"])

        pred_gnn, pure_gnn = gnn_result if gnn_result else (None, [None]*len(mix["smiles"]))
        pred_ct,  pure_ct  = ct_result  if ct_result  else (None, [None]*len(mix["smiles"]))

        if pred_gnn is not None and pred_ct is not None:
            pred_ens = (pred_gnn + pred_ct) / 2
        else:
            pred_ens = pred_gnn if pred_gnn is not None else pred_ct

        results.append({
            **mix,
            "pred_gnn": pred_gnn,
            "pred_ct":  pred_ct,
            "pred_ens": pred_ens,
            "pure_gnn": pure_gnn,
            "pure_ct":  pure_ct,
        })

    print_per_row(results)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    for ref_key, ref_label in [("ysi_meas", "Measured"), ("ysi_reg", "Regression")]:
        subset = [r for r in results if r[ref_key] is not None]
        if not subset:
            continue
        actual  = [r[ref_key]    for r in subset]
        pred_a  = [r["pred_gnn"] for r in subset]
        pred_b  = [r["pred_ct"]  for r in subset]
        pred_c  = [r["pred_ens"] for r in subset]
        valid_a = [(a, p) for a, p in zip(actual, pred_a) if p is not None]
        valid_b = [(a, p) for a, p in zip(actual, pred_b) if p is not None]
        valid_c = [(a, p) for a, p in zip(actual, pred_c) if p is not None]

        print(f"vs {ref_label}:")
        if valid_a:
            aa, pa = zip(*valid_a)
            mae, rmse, r2 = metrics_str(list(aa), list(pa))
            print(f"  Method A (GNN)         n={len(valid_a):2d}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
        if valid_b:
            ab, pb = zip(*valid_b)
            mae, rmse, r2 = metrics_str(list(ab), list(pb))
            print(f"  Method B (carbon type) n={len(valid_b):2d}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
        if valid_c:
            ac, pc = zip(*valid_c)
            mae, rmse, r2 = metrics_str(list(ac), list(pc))
            print(f"  Method C (ensemble)    n={len(valid_c):2d}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
        print()

    # ── Plot ───────────────────────────────────────────────────────────────────
    ref_sets = [(k, l) for k, l in [("ysi_meas","Measured"),("ysi_reg","Regression")]
                if any(r[k] for r in results)]

    n_cols   = len(ref_sets)
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 18))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    method_rows = [
        ("pred_gnn", "Method A — GNN predictor"),
        ("pred_ct",  "Method B — Carbon types (Eq 7b)"),
        ("pred_ens", "Method C — Ensemble (mean of A & B)"),
    ]

    for col, (ref_key, ref_label) in enumerate(ref_sets):
        subset = [r for r in results if r[ref_key] is not None]
        actual = [r[ref_key] for r in subset]
        flags  = [r["complete"] for r in subset]

        for row, (pred_key, m_label) in enumerate(method_rows):
            pred = [r[pred_key] for r in subset]
            valid = [(a, p, c) for a, p, c in zip(actual, pred, flags) if p is not None]
            if not valid:
                axes[row, col].set_visible(False)
                continue
            av, pv, cv = zip(*valid)
            make_panel(axes[row, col], list(av), list(pv), list(cv), m_label, ref_label)

    plt.suptitle("Mixture YSI: Predicted vs Reference", fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = RESULTS_DIR / "mixture_ysi_actual_vs_pred.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path.relative_to(ROOT)}")
    plt.close()


if __name__ == "__main__":
    main()
