import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. Define transforms
tranform = transforms.ToTensor()

# 2. Download and load the training data
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=tranform,
    download=True
)

# 3. Create DataLoader for batching.
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# Step 4: Loop through DataLoader
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx} - Image batch shape: {images.shape} | Label batch shape: {labels.shape}")
    if batch_idx == 2:
        break

