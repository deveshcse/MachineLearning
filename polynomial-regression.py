# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Read the dataset from a CSV file
data = pd.read_csv(r'dataset/data.csv')
print(data.head())  # Display the first few rows of the dataset

# Extract input (X) and output (y) variables
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# Create a Linear Regression model and fit it to the data
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Create a Polynomial Regression model with degree=4 and fit it to the data
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)

# Create a Linear Regression model for the polynomial features and fit it to the data
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualize the Linear and Polynomial Regression results
plt.scatter(X, y, color='blue')  # Plot the original data points

# Plot the Linear Regression line
plt.plot(X, lin_reg.predict(X), color='red', label='Linear Regression')

# Plot the Polynomial Regression line
plt.plot(X, lin_reg2.predict(poly.fit_transform(X)), color='green', label='Polynomial Regression')

# Add title and labels to the plot
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

# Display legend to distinguish between Linear and Polynomial Regression
plt.legend()

# Show the plot
plt.show()
