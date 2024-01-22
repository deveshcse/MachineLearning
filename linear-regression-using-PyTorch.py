import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
torch.manual_seed(42)
X = torch.rand(100, 1)
y = 3 * X + 2 + 0.1 * torch.randn(100, 1)


# Define a linear regression model using PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Instantiate the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Convert predictions to numpy for plotting
predictions = predictions.detach().numpy()

# Plot the data points and regression line
plt.scatter(X.numpy(), y.numpy(), label='Data points')
plt.plot(X.numpy(), predictions, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using PyTorch')
plt.legend()
plt.show()
