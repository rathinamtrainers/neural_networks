import torch

x = torch.tensor([2.0], requires_grad=True)

y = x.detach()

with torch.no_grad():
    z = x * 3

print(f"x: {x}")
print(f"y: {y}")
print(f"z: {z}")