# Import the required modules
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the diabetes dataset
from sklearn.pipeline import Pipeline

# Load the diabetes dataset using the load_diabetes function
X, y = load_diabetes(return_X_y=True)

# Create a pipeline with PCA (Principal Component Analysis) and linear regression
pca = PCA(n_components=5)  # Set the number of principal components to retain

reg = LinearRegression()  # Create a linear regression model
pipeline = Pipeline(steps=[('pca', pca),
                           ('reg', reg)])  # Combine PCA and linear regression in a pipeline

# Fit the pipeline to the data
pipeline.fit(X, y)

# Predict the labels for the data
y_pred = pipeline.predict(X)

# Compute the evaluation metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = pipeline.score(X, y)

# Print the number of features before and after PCA
print(f'Number of features before PCA: {X.shape[1]}')
print(f'Number of features after PCA: {pca.n_components_}')

# Print the evaluation metrics
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R^2: {r2:.2f}')
