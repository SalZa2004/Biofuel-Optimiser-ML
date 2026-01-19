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

