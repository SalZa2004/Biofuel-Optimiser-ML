from flask import Flask, render_template, request, redirect, url_for, send_file, session
import pandas as pd
import os
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.base import BaseEstimator, RegressorMixin

# -------------------------
# CLEAN CN PREDICTOR CLASS 
# -------------------------
class CleanCNPredictor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model_type = None
        self.model_params = None
        self.selector_params = None
        self.trained_model = None
        
    def set_model(self, model):
        self.trained_model = model
        
    def set_selector(self, selector):
        self.selector_params = {
            'n_morgan': int(selector.n_morgan),
            'corr_cols_to_drop': list(selector.corr_cols_to_drop),
            'selected_indices': selector.selected_indices.tolist() if hasattr(selector.selected_indices, 'tolist') else list(selector.selected_indices)
        }
    
    def _get_descriptor_functions(self):
        desc_list = Descriptors._descList
        return [d[1] for d in desc_list]
    
    def _morgan_fp_from_mol(self, mol, radius=2, n_bits=2048):
        from rdkit.Chem import rdFingerprintGenerator
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = fpgen.GetFingerprint(mol)
        return np.array(list(fp.ToBitString()), dtype=int)
    
    def _physchem_desc_from_mol(self, mol):
        try:
            descriptor_functions = self._get_descriptor_functions()
            desc = np.array([fn(mol) for fn in descriptor_functions], dtype=np.float32)
            return np.nan_to_num(desc)
        except:
            return None
    
    def _featurize_single(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fp = self._morgan_fp_from_mol(mol)
        desc = self._physchem_desc_from_mol(mol)
        if fp is None or desc is None:
            return None
        
        return np.hstack([fp, desc])
    
    def _apply_selection(self, X):
        n_morgan = self.selector_params['n_morgan']
        X_mfp = X[:, :n_morgan]
        X_desc = X[:, n_morgan:]
        
        desc_df = pd.DataFrame(X_desc)
        corr_cols = self.selector_params['corr_cols_to_drop']
        desc_filtered = desc_df.drop(columns=corr_cols, axis=1).values
        
        X_corr = np.hstack([X_mfp, desc_filtered])
        selected_indices = self.selector_params['selected_indices']
        return X_corr[:, selected_indices]
    
    def featurize(self, smiles_input):
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
        else:
            smiles_list = smiles_input
        
        features = []
        for smi in smiles_list:
            fv = self._featurize_single(smi)
            if fv is not None:
                features.append(fv)
        
        if len(features) == 0:
            raise ValueError("No valid SMILES strings provided!")
        
        return np.vstack(features)
    
    def predict(self, smiles_input):
        X_full = self.featurize(smiles_input)
        X_sel = self._apply_selection(X_full)
        return self.trained_model.predict(X_sel)
    
    def predict_single(self, smiles):
        return self.predict([smiles])[0]

# -------------------------
# LOAD MODEL
# -------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATHS = [
    os.path.join(MODEL_DIR, "cn_predictor_clean.pkl"),
    os.path.join(MODEL_DIR, "cn_predictor_complete.pkl")
]

def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                print(f"Model loaded: {path}")
                return model
            except Exception as e:
                print(f"Failed loading {path}: {e}")
    print("❌ Could not load any model file.")
    return None

model = load_model()   # <-- FIXED (this was missing!)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def validate_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

def predict_cetane(smiles):
    if not validate_smiles(smiles):
        return None
    try:
        return float(model.predict_single(smiles))
    except:
        return None

import pubchempy as pcp

def pubchem_name_to_smiles(name):
    """Return canonical SMILES from a compound name."""
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if name == "":
        return None

    try:
        results = pcp.get_compounds(name, "name")
        if not results:
            return None
        return results[0].canonical_smiles
    except Exception:
        return None


def pubchem_smiles_to_name(smiles):
    """Return preferred IUPAC name from SMILES."""
    try:
        results = pcp.get_compounds(smiles, "smiles")
        if not results:
            return None
        compound = results[0]

        # Prefer IUPAC name if available
        if getattr(compound, "iupac_name", None):
            return compound.iupac_name

        # Fallback to title
        return compound.title
    except Exception:
        return None

# Run Flask app
app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/pure", methods=["GET", "POST"])
def pure_predictor():
    results = []
    error = None

    #------------------
    # CSV File
    #------------------
    if request.method == "POST" and request.form.get("mode") == "csv":
        csv_file = request.files.get("csv_file")

        if not csv_file:
            error = "No CSV file uploaded."
            return render_template("pure_predictor.html", results=results, error=error)

        try:
            df = pd.read_csv(csv_file)

            if "SMILES" not in df.columns:
                error = "CSV must contain a 'SMILES' column."
                return render_template("pure_predictor.html", results=results, error=error)

            for i, row in df.iterrows():
                raw_name = row.get("IUPAC names", "")
                if pd.isna(raw_name):
                    name = ""
                else:
                    name = str(raw_name).strip()

                raw_smiles = row.get("SMILES", "")
                if pd.isna(raw_smiles):
                    smiles = ""
                else:
                    smiles = str(raw_smiles).strip()

                entry = {
                    "name": name if name else "-",
                    "smiles": smiles,
                    "dcn": None,
                    "error": None,
                    "img_id": None
                }

            
                # STEP 1 — If SMILES empty → convert NAME → SMILES

                if smiles == "" and name not in ("", None, "-"):
                    final_smiles = pubchem_name_to_smiles(name)
                    if final_smiles is None:
                        entry["error"] = "Name not found in PubChem"
                        results.append(entry)
                        continue
                else:
                    final_smiles = smiles

               
                # STEP 2 — Validate SMILES
                
                if not validate_smiles(final_smiles):
                    entry["error"] = "Invalid SMILES"
                    results.append(entry)
                    continue

                entry["smiles"] = final_smiles

                
                # STEP 3 — Convert SMILES → IUPAC name
                
                iupac_name = pubchem_smiles_to_name(final_smiles)
                if (not name or name == "-") and iupac_name:
                    entry["name"] = iupac_name

                
                # STEP 4 — Predict DCN
                
                pred = predict_cetane(final_smiles)
                if pred is None:
                    entry["error"] = "Prediction failed"
                else:
                    entry["dcn"] = round(pred, 2)

                    mol = Chem.MolFromSmiles(final_smiles)
                    img = Draw.MolToImage(mol, size=(300, 250))

                    img_filename = f"mol_csv_{i}.png"
                    img_path = os.path.join("app", "static", "generated", img_filename)
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    img.save(img_path)

                    entry["img_id"] = img_filename

                results.append(entry)

            return render_template("pure_predictor.html", results=results)

        except Exception as e:
            error = f"Failed to read CSV file: {e}"
            return render_template("pure_predictor.html", results=results, error=error)

    #------------------
    # Manual input 
    #------------------
    elif request.method == "POST":
        names = request.form.getlist("fuel_name[]")
        smiles_list = request.form.getlist("smiles[]")

        for i, (name, smiles) in enumerate(zip(names, smiles_list)):
            name = name.strip()
            smiles = smiles.strip()

            entry = {
                "name": name if name else "-",
                "smiles": smiles,
                "dcn": None,
                "error": None,
                "img_id": None
            }

            
            # STEP 1 — If SMILES empty → convert NAME → SMILES
            
            if smiles == "" and name not in ("", None, "-"):
                final_smiles = pubchem_name_to_smiles(name)
                if final_smiles is None:
                    entry["error"] = "Name not found in PubChem"
                    results.append(entry)
                    continue
            else:
                final_smiles = smiles

            
            # STEP 2 — Validate SMILES
            
            if not validate_smiles(final_smiles):
                entry["error"] = "Invalid SMILES"
                results.append(entry)
                continue

            entry["smiles"] = final_smiles

            
            # STEP 3 — Convert SMILES → IUPAC name
            
            iupac_name = pubchem_smiles_to_name(final_smiles)
            if (not name or name == "-") and iupac_name:
                entry["name"] = iupac_name

            
            # STEP 4 — Predict & draw molecule
            
            pred = predict_cetane(final_smiles)
            if pred is None:
                entry["error"] = "Prediction failed"
            else:
                entry["dcn"] = round(pred, 2)

                mol = Chem.MolFromSmiles(final_smiles)
                img = Draw.MolToImage(mol, size=(300, 250))

                img_filename = f"mol_{i}.png"
                img_path = os.path.join("app", "static", "generated", img_filename)
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                img.save(img_path)

                entry["img_id"] = img_filename

            results.append(entry)

    return render_template("pure_predictor.html", results=results, error=error)


@app.route("/download_results", methods=["POST"])
def download_results():
    import io, json
    results_json = request.form.get("results_data")
    results = json.loads(results_json)

    cleaned_rows = []

    for r in results:
        cleaned_rows.append({
            "IUPAC Name": r.get("name", "-"),
            "SMILES": r.get("smiles", "-"),
            "Predicted DCN": r.get("dcn", None),
            "Status": ("OK" if r.get("error") in (None, "", "OK") else r.get("error"))
        })

    df = pd.DataFrame(cleaned_rows)

    # column order
    df = df[["IUPAC Name", "SMILES", "Predicted DCN", "Status"]]

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="pure_fuel_predictions.csv"
    )


@app.route("/mixture")
def mixture_predictor():
    return render_template("mixture_predictor.html")

@app.route("/generate")
def generative():
    return render_template("generative.html")

@app.route("/constraints")
def constraints():
    return render_template("constraints.html")

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

@app.route("/download/pure")
def download_pure():
    return send_file(
        "datasets/pure_database.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="pure_fuel_dataset.xlsx"
    )

@app.route("/download/mixture")
def download_mixture():
    return send_file(
        "datasets/mixture_database.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="mixture_fuel_dataset.xlsx"
    )

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
