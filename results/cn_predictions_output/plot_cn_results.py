import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("cn_predictions.csv")

# Extract columns
y_true = df["CN_Measured"]
y_pred = df["CN_predicted"]
residuals = y_true - y_pred
plt.figure()
plt.scatter(y_true, y_pred)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()])

plt.xlabel("Measured Cetane Number")
plt.ylabel("Predicted Cetane Number")
plt.title("Measured vs Predicted Cetane Number")
plt.show()
plt.figure()
plt.scatter(y_true, residuals)
plt.axhline(0)

plt.xlabel("Measured Cetane Number")
plt.ylabel("Residual (Measured − Predicted)")
plt.title("Residual Plot")
plt.show()
y_std = df["CN_pred_std"]

plt.figure()
plt.errorbar(y_true, y_pred, yerr=y_std, fmt='o')
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()])

plt.xlabel("Measured Cetane Number")
plt.ylabel("Predicted Cetane Number")
plt.title("Measured vs Predicted CN with Uncertainty")
plt.show()
