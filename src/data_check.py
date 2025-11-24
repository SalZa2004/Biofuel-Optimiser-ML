from feature_selection import get_selected_data
import numpy as np
X_train, y_train = get_selected_data()

# Check original values
y_original = np.expm1(y_train)
print(f"Original CN range: {y_original.min():.1f} - {y_original.max():.1f}")
print(f"Original CN mean: {y_original.mean():.1f}")
print(f"Original CN std: {y_original.std():.1f}")