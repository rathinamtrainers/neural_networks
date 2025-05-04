import torch
import torch.nn as nn
import torch.optim as optim

# Create synthetic data
X = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)
y = 2 * X + 1 + torch.rand(X.size()) * 0.8  # Add noise to simulate real-world imperfect data.


# Define linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

# Initialize model, loss function and optimizer
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{300}], Loss: {loss.item():.4f}")

# Test the model
model.eval()
prediction = model(X).detach()

print(f"Input: {X[0].item():.4f}, Output: {prediction[0].item():.4f}")

# Plot the results
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.scatter(X.numpy(), y.numpy(), label='Original Data')
plt.plot(X.numpy(), prediction.numpy(), 'r-', label='Fitted line')
plt.legend()
plt.title('Linear Regression with PyTorch')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
