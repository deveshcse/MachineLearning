import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng()

# Generate some data:
x = rng.random(10)
y = 1.6 * x + rng.random(10)

# Perform the linear regression:
res = stats.linregress(x, y)

'''The coefficient of determination (R²) is a number between 0 and 1 that measures how well a statistical model
 predicts an outcome. You can interpret the R² as the proportion of variation in the dependent variable
  that is predicted by the statistical model.'''

# Coefficient of determination (R-squared):
print(f"R-squared: {res.rvalue ** 2:.6f}")
# Slope and Intercept
print("slope", res.slope)
print("Intercept", res.intercept)

# Plot the data along with the fitted line:
plt.plot(x, y, 'o', label='original data')
plt.plot(x, res.intercept + res.slope * x, 'r', label='fitted line')
plt.legend()
plt.show()
