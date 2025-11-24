from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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
    
    mae_train = mean_absolute_error(np.expm1(y_train), np.expm1(pred_train))
    mae_test = mean_absolute_error(np.expm1(y_test), np.expm1(pred_test))
    
    rmse_train = root_mean_squared_error(np.expm1(y_train), np.expm1(pred_train))
    rmse_test = root_mean_squared_error(np.expm1(y_test), np.expm1(pred_test))

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

# Enhanced MLP
models ={
    # Try adding more regularization
    'MLP_Enhanced_v2' : MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    alpha=0.01,  # Increase regularization
    learning_rate='adaptive',
    early_stopping=True,  # Add early stopping
    validation_fraction=0.1,
    n_iter_no_change=20,
    max_iter=1000,
    random_state=42
    )
}
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    results[name] = evaluate(model, X_train, y_train, X_test, y_test)

df_results = pd.DataFrame(results).T
print(df_results)

