from sklearn.ensemble import IsolationForest
from core.data_prep import df
class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)
    

    def filter_anomalies(self, df):
        """Filter out anomalies from the DataFrame."""
        if df.empty:
            return df  # Return empty DataFrames if input is empty
        
        features = df.select_dtypes  # Use only numeric features
        if features.empty:
            return df  # No numeric features to analyze
        
        self.fit(features)
        predictions = self.predict(features)
        
        # -1 indicates anomaly, 1 indicates normal
        normal_df = df[predictions == 1].copy()
        anomalies_df = df[predictions == -1].copy()
        
        return normal_df, anomalies_df