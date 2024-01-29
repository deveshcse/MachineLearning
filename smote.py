import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter

os.environ['LOKY_MAX_CPU_COUNT'] = '6'  # Set the number of cores you want to use
# Generate a synthetic imbalanced dataset
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=42)

# Convert y to strings for seaborn countplot
y_str = y.astype(str)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot for the original dataset
sns.countplot(x=y_str, ax=axes[0], order=['0', '1'])
axes[0].set_title("Original Imbalanced Dataset")

# Print the class distribution before SMOTE
print("Class distribution before SMOTE:", Counter(y))

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Count plot for the resampled dataset after SMOTE
sns.countplot(x=y_resampled.astype(str), ax=axes[1], order=['0', '1'])
axes[1].set_title("Resampled Dataset After SMOTE")

# Print the class distribution after SMOTE
print("Class distribution after SMOTE:", Counter(y_resampled))

# Show the subplots
plt.show()
