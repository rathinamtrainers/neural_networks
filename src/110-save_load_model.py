import torch
import torch.nn as nn
import torch.optim as optim
import os

# Step-1: Define the model using nn.Module
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


# Step-2: Create model instance and move to GPU/CPU device
model = LinearRegression()

# Step-3: Create synthetic data
X = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[3.0], [5.0], [7.0]])

# Step-4: Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step-5: Training loop
for epoch in range(100):
    model.train()
    pred_y = model(X)   # Forward pass
    loss = criterion(pred_y, y)
    optimizer.zero_grad()   # Step-1: Clear gradients
    loss.backward()         # Step-2: Compute gradients (Backward pass)
    optimizer.step()        # Step-3: Update weights

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{300}], Loss: {loss.item():.4f}")


# Save only weights
torch.save(model.state_dict(), "model_weights.pth")
print("Saved PyTorch Model State to model_weights.pth")

# save entire model
torch.save(model, "model.pth")
print("Saved PyTorch Model to model.pth")

# Save model weights and optimizer state
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "checkpoint.pth")
print("Saved PyTorch Model State to checkpoint.pth")
