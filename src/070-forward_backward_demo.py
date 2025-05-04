import torch
import torch.nn as nn
import torch.optim as optim

# Input and labels
x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=False)
y = torch.tensor([[3.0], [5.0], [7.0]], requires_grad=False)    # y = 2x + 1

# Simple model with 1 neuron
model = nn.Linear(1, 1)
print(f"Initial weights: {model.weight.item()}, bias: {model.bias.item()}")

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward pass
pred = model(x)
loss = criterion(pred, y)
print(f"Loss after forward pass: {loss.item()}")

# Backward pass
optimizer.zero_grad()    # Step-1: Clear gradients
loss.backward()          # Step-2: Compute gradients (Backward pass)
optimizer.step()         # Step-3: Update weights
print(f"Updated weights: {model.weight.item()}, bias: {model.bias.item()}")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Gradients: {param.grad}")


