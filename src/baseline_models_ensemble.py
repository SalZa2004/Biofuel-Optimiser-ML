from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Optional models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from feature_engineering import get_training_data

X, y = get_training_data()

def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_test = mean_absolute_error(y_test, pred_test)
    
    rmse_train = root_mean_squared_error(y_train, pred_train)
    rmse_test = root_mean_squared_error(y_test, pred_test)

    return {
        "MAE_train": mae_train,
        "MAE_test": mae_test,
        "RMSE_train": rmse_train,
        "RMSE_test": rmse_test
    }


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


n_fp = 2048
n_desc = X.shape[1] - n_fp

scaler = StandardScaler()

# Fit only on training descriptor columns
X_train[:, n_fp:] = scaler.fit_transform(X_train[:, n_fp:])
X_test[:, n_fp:] = scaler.transform(X_test[:, n_fp:])

models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
}
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    results[name] = evaluate(model, X_train, y_train, X_test, y_test)

df_results = pd.DataFrame(results).T
print(df_results)

# Save results to CSV
df_results.to_csv("baseline_model_results.csv")