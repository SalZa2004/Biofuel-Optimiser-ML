import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from feature_selection import get_selected_data
import pandas as pd
X, y = get_selected_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load('extratrees_model.pkl')
test_pred = model.predict(X_test)

test_rmse = np.sqrt(np.mean((y_test - test_pred)**2))
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test MAE:  {test_mae:.2f}")
print(f"Test R²:   {test_r2:.3f}")
print(f"% Error:   {(test_rmse/y_test.mean())*100:.1f}%")

results =pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R2", "%\ Error"],
    "Value": [test_rmse, test_mae, test_r2, (test_rmse/y_test.mean())*100]
}).set_index("Metric")
# Save results to CSV
results.to_csv("ExtraTrees_Performance.csv")