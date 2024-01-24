import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Function to generate synthetic data with noise
def generate_data(n, true_slope, true_intercept, cauchy_scale):
    # Generate n equally spaced points between 0 and 10 as x values
    x = np.linspace(0, 10, n)

    # Create the true line without noise based on slope and intercept
    true_line = true_slope * x + true_intercept

    # Generate Cauchy-distributed noise with scale specified by cauchy_scale
    cauchy_noise = np.random.standard_cauchy(n) * cauchy_scale

    # Add the noise to the true line to get the synthetic y values
    y = true_line + cauchy_noise
    return x, y


# Function to fit a linear model using least squares
def fit_least_squares(x, y):
    # Create a matrix 'a' where each row is [x_value, 1]
    a = np.vstack([x, np.ones_like(x)]).T

    # Use least squares to find the coefficients (slope and intercept)
    # The result of np.linalg.lstsq is a tuple, and [0] extracts the solution
    slope, intercept = np.linalg.lstsq(a, y, rcond=None)[0]
    return slope, intercept


# Function to fit quantile regression for specified quantile
def fit_quantile_regression(x, y, quantile):
    # Create a QuantReg model with the input x and y values
    quantile_reg = sm.QuantReg(y, sm.add_constant(x))

    # Fit the quantile regression model using the specified quantile
    result = quantile_reg.fit(q=quantile)

    # Return the parameters (coefficients) of the fitted quantile regression model
    return result.params

# Function to plot the results
def plot():
    # Create a figure with specified size
    plt.figure(figsize=(10, 5))

    # Scatter plot of the data with Cauchy noise
    plt.scatter(x, y, label='Data with Cauchy noise', alpha=0.5, s=6)

    # Plot the true line without noise
    plt.plot(x, true_slope * x + true_intercept, color='red', label='True Line')

    # Plot the linear regression line (OLS)
    plt.plot(x, ls_slope * x + ls_intercept, color='green', label='Linear Regression (Mean, OLS)')

    # Plot quantile regression lines for quartiles 0.25, 0.5, and 0.75
    plt.plot(x, quantile_slope_25 * x + quantile_intercept_25, color='purple', label='Quantile Regression (Q=0.25)')
    plt.plot(x, quantile_slope_50 * x + quantile_intercept_50, color='blue', label='Quantile Regression (Q=0.5)')
    plt.plot(x, quantile_slope_75 * x + quantile_intercept_75, color='orange', label='Quantile Regression (Q=0.75)')
    plt.legend()
    # Set y-axis limit for better visualization
    plt.ylim(-10, 40)
    plt.title('OLS vs QR Fit with Cauchy Noise')
    plt.show()


# Set true parameters
true_slope = 2
true_intercept = 5
cauchy_scale = 3

# Generate data
x, y = generate_data(1000, true_slope, true_intercept, cauchy_scale)

# Fit a linear model using least squares
ls_slope, ls_intercept = fit_least_squares(x, y)

# Fit quantile regression for different quartiles
quantile_intercept_25, quantile_slope_25 = fit_quantile_regression(x, y, 0.25)
quantile_intercept_50, quantile_slope_50 = fit_quantile_regression(x, y, 0.5)
quantile_intercept_75, quantile_slope_75 = fit_quantile_regression(x, y, 0.75)

# Plot the results
plot()

