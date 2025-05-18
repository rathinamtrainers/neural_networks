import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

# 2. Download and load the training data
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# 3. Create DataLoader for batching.
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# 4. Inspect a sample.
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(f"Batch Index: {batch_idx}")
print(f"Image batch shape: {example_data.shape}")
print(f"Target batch shape: {example_targets.shape}")

# 5. Visualize a few images.
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap="gray")
    plt.title(f"Label: {example_targets[i].item()}")
    plt.axis("off")

plt.suptitle("MNIST Sample Images")
plt.show()

