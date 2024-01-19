import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# Create a Linear Regression object
reg = LinearRegression()

# Train the model
reg.fit(X, y)

# coefficient of determination
print("coefficient of determination is:", reg.score(X, y))
print(reg.coef_)
print( reg.intercept_)

# Plot the regression line
plt.scatter(X, y, color='black')
plt.plot(X, reg.predict(X), color='red', linewidth=3)
plt.title('Linear Regression with Scikit learn')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
