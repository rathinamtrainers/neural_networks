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
checkpoint = torch.load("model_weights.pth")

model = LinearRegression()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

optimizer_state_dict = checkpoint['optimizer_state_dict']
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.load_state_dict(optimizer_state_dict)



with torch.no_grad():
    test_input = torch.tensor([[10.0]])
    pred_test = model(test_input)
    print(f"Input: {test_input.item():.4f}, Output: {pred_test.item():.4f}")


