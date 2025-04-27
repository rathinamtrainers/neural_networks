import torch

a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

b = a * 2

c = b.mean()

# Compute gradients
c.backward()

# Print gradients
print(f"a: {a}")
print(f"b: {b}")
print(f"c: {c}")
print(f"dx/da: {a.grad}")

