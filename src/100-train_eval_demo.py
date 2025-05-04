import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn

# 1. Generate synthetic data
X = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)
y = 2 * X + 1 + torch.rand(X.size()) * 0.8  # y = 2X + 1 + noise

# 2. Train-test split (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to float tensors.
X_train, X_test, y_train, y_test = X_train.float(), X_test.float(), y_train.float(), y_test.float()


# 3. Define model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
train_losses = []
for epoch in range(200):
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{300}], Loss: {loss.item():.4f}")


# 5. Test the model
model.eval()
with torch.no_grad():
    pred_test = model(X_test)
    test_loss = mean_squared_error(y_test.numpy(), pred_test.numpy())
    r2  = r2_score(y_test.numpy(), pred_test.numpy())

print(f"Test Loss: {test_loss:.4f}")
print(f"R2 score: {r2:.4f}")

# Plot the Training loss curve
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
