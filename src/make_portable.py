# Load and predict
from portable_model import CNPredictorPortable

# Load and predict
model_path = "cn_predictor_complete.pkl"
predictor = CNPredictorPortable.load(model_path)
cn = predictor.predict_single("CCCCCCCO") #
print(f"Cetane Number: {cn:.2f}")