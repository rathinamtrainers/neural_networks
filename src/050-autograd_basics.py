import torch

# Create a tensor with requires_grad=True
x = torch.tensor([2.0], requires_grad=True)

# Define a simple function: y = x^2 + 3x + 4
y = x ** 2 + 3 * x + 4

# Compute gradient.
y.backward()

# Print gradient dy/dx
print(f"x: {x}")
print(f"y: {y}")
print(f"dx/dy: {x.grad}")