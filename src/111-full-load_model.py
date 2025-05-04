import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

# Load the full model
model = torch.load("model.pth", weights_only=False)

model.eval()

with torch.no_grad():
    test_input = torch.tensor([[10.0]])
    pred_test = model(test_input)
    print(f"Input: {test_input.item():.4f}, Output: {pred_test.item():.4f}")


