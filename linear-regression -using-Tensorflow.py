import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set the random seed
tf.random.set_seed(42)

# Generate random data
X = tf.constant(np.random.rand(100, 1), dtype=tf.float32)
y = tf.constant(3 * X + 2 + 0.1 * np.random.randn(100, 1), dtype=tf.float32)


# Define a linear regression model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1, activation='linear')
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, verbose=0)

# Make predictions
predictions = model.predict(X)

# Plot the data points and regression line
plt.scatter(X, y, label='Data points')
plt.plot(X, predictions, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using TensorFlow')
plt.legend()
plt.show()
