import torch
import torch.nn as nn
import torch.optim as optim

# Step-1: Define the model using nn.Module
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.output = nn.Linear(10, 1)

    def forward(self, x):           # x is the input
        x = self.hidden(x)          # Hidden layer processes the input
        x = self.relu(x)            # Activation function (relu) is applied.
        x = self.output(x)          # Output layer
        return x                    # returns the result of the output layer


# Step-2: Create model instance and move to GPU/CPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyNeuralNetwork().to(device)

# Step-3: Create synthetic data (y = 3x + 2)
X = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1).to(device)
y = 3 * X + 2 + torch.rand(X.size()).to(device) * 0.5  # Add noise to simulate real-world imperfect data.

# Step-4: Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step-5: Train the model.
for epoch in range(300):
    model.train()
    pred_y = model(X)   # Forward pass
    loss = criterion(pred_y, y)
    optimizer.zero_grad()   # Step-1: Clear gradients
    loss.backward()         # Step-2: Compute gradients (Backward pass)
    optimizer.step()        # Step-3: Update weights

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{200}], Loss: {loss.item():.4f}")

model.eval()
test_input = torch.tensor([[10.0]], device=device)
test_ouput = model(test_input)
print(f"Input: {test_input.item():.4f}, Output: {test_ouput.item():.4f}")



