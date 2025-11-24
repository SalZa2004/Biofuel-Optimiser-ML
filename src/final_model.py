from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import numpy as np
from feature_selection import get_selected_data
X, y = get_selected_data()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Remove invalid and extreme values

# Best parameters from tuning

best_params = {"num_leaves": 80,
  "max_depth": 9,
  "learning_rate": 0.032756728584024876,
 "min_child_samples": 28,
  "subsample": 0.6754828921395883,
  "colsample_bytree": 0.514886954190426,
  "reg_alpha": 0.03704108745886531,
  "reg_lambda": 1.4740784771442335}

final_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)


final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(period=20)
    ]
)
from sklearn.metrics import mean_squared_error
# evaluate on test (remember to inverse log if needed)

y_pred = final_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_train = np.sqrt(mean_squared_error(y_train, final_model.predict(X_train)))
print("Test RMSE:", rmse)
print("Train RMSE:", rmse_train)
print("Percentage error :", (rmse / (y_test.max() - y_test.min())) * 100)

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# learning curve (use log target if training in log)
train_sizes, train_scores, val_scores = learning_curve(
    final_model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1,1.0,6), scoring='neg_root_mean_squared_error', n_jobs=-1
)
train_scores = -train_scores; val_scores = -val_scores

plt.plot(train_sizes, train_scores.mean(axis=1), label='train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='test')
plt.xlabel('Train set size'); plt.ylabel('RMSE (log)')
plt.legend(); plt.show()
