import torch

x = torch.tensor([5.0], requires_grad=True)
lr = 0.1

for i in range(20):
    y = x ** 2
    y.backward()

    with torch.no_grad():
        x.data = x.data - lr * x.grad     # x = x - learning rate x gradient (Gradient Descent Formula)
        x.grad.zero_()

    print(f"Step {i + 1}: x = {x.item():.4f}, y = {y.item():.4f}")
