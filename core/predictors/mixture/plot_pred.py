import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/cn_predictions_output/cn_predictions.csv')
error = df['CN_Measured'] - df['CN_predicted']
rmse = (error ** 2).mean() ** 0.5
print(f'RMSE: {rmse}')
print(df.head())
r2_score = 1 - (error ** 2).sum() / ((df['CN_Measured'] - df['CN_Measured'].mean()) ** 2).sum()
plt.figure(figsize=(8, 8))
plt.scatter(df['CN_Measured'], df['CN_predicted'])
plt.plot([0, 100], [0, 100], 'r--')
plt.xlabel('Measured CN')
plt.ylabel('Predicted CN')
plt.title('Measured vs Predicted CN')
plt.legend([f'RMSE: {rmse:.2f}'])
plt.xlim(0, 100)
plt.ylim(0, 120)
plt.grid()
plt.show()

def outliers(df, threshold=10):
    error = df['CN_Measured'] - df['CN_predicted']
    outlier_indices = error[error.abs() > threshold].index
    return df.loc[outlier_indices]
outlier_df = outliers(df)
print("Outliers:")
print(outlier_df)
save_path = 'results/cn_predictions_output/cn_outliers.csv'
outlier_df.to_csv(save_path, index=False)

residual = df['CN_Measured'] - df['CN_predicted']
plt.figure(figsize=(8, 6))
plt.scatter(df['CN_Measured'], residual)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Measured CN')
plt.ylabel('Residuals (Measured - Predicted)')
plt.title('Residuals vs Measured CN')
plt.grid()
plt.show()