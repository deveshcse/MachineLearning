import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv(r'dataset/data.csv')
print(data.head())

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualising the Linear and Polynomial Regression results
plt.scatter(X, y, color='blue')

plt.plot(X, lin_reg.predict(X), color='red', label='Linear Regression')
plt.plot(X, lin_reg2.predict(poly.fit_transform(X)), color='green', label='Polynomial Regression')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.legend()
plt.ylabel('Pressure')
plt.show()
