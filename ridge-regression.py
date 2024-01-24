import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Ridge regression
alpha = 1.0  # Regularization strength (adjust as needed)
ridge_reg = Ridge(alpha=alpha)
ridge_reg.fit(X_train, y_train)

# Display the coefficients and intercept
print("Ridge Coefficients:", ridge_reg.coef_)
print("Ridge Intercept:", ridge_reg.intercept_)

# Make predictions on the test set
y_pred = ridge_reg.predict(X_test)

# Plot the results
plt.scatter(X_test, y_test, label='Test Data', color='blue')
plt.plot(X_test, y_pred, label='Ridge Regression', color='red')
plt.title('Ridge Regression Example')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
