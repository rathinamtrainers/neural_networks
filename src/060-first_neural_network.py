import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create synthetic data: y = 2x + 1
X = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1).to(device)
y = 2 * X + 1 + torch.rand(X.size()).to(device)  # Add noise to simulate real-world imperfect data.

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# Instantiate the model
model = SimpleNN().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(300):
    model.train()

    # Forward pass
    pred = model(X)
    loss = criterion(pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for monitoring
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{300}], Loss: {loss.item():.4f}")

# Test the model
model.eval()
test_x = torch.tensor([[6.0], [10.0]], device=device)
pred_y = model(test_x)

print(f"Predicted y for x=6: {pred_y[0].item():.4f}")
print(f"Predicted y for x=10: {pred_y[1].item():.4f}")



