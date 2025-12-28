from pathlib import Path
from huggingface_hub import snapshot_download

HF_MODELS = {
    "cn": "SalZa2004/Cetane_Number_Predictor",
    "ysi": "SalZa2004/YSI_Predictor",
    "bp": "SalZa2004/Boiling_Point_Predictor",
    "density": "SalZa2004/Density_Predictor",
    "lhv": "SalZa2004/LHV_Predictor",
    "dynamic_viscosity": "SalZa2004/Dynamic_Viscosity_Predictor",
}

def load_models():
    return {
        k: Path(snapshot_download(repo_id=v, repo_type="model"))
        for k, v in HF_MODELS.items()
    }
