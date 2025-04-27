import torch

x = torch.tensor([2.0], requires_grad=True)

y = x * 3
y.backward()
print(f"x: {x}")
print(f"y: {y}")
print(f"First Backward: dx/dy: {x.grad}")

y = x * 3
y.backward()
print(f"x: {x}")
print(f"y: {y}")
print(f"Second Backward: dx/dy: {x.grad}")

# Reset gradients
x.grad.zero_()
print(f"After Zeroing: {x.grad}")

