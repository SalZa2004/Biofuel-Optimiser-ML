from huggingface_hub import hf_hub_download
import joblib
from shared_features import featurize_df, FeatureSelector

repo = "SalZa2004/Cetane_Number_Predictor"

model_path = hf_hub_download(repo, "model.joblib")
selector_path = hf_hub_download(repo, "selector.joblib")

model = joblib.load(model_path)
selector = joblib.load(selector_path)

def predict_ysi(smiles):
    X = featurize_df([smiles], return_df=False)  
    X = selector.transform(X)
    return float(model.predict(X)[0])

print(predict_ysi("CCCCCCC"))
