import torch

# Check if GPU is available
device_to_use = "cpu"
if torch.cuda.is_available():
    device_to_use = "cuda"
device = torch.device(device_to_use)
print(f"Using device: {device}")

# Create two tensors.
a = torch.tensor([2.0, 3.0])
b = torch.tensor([6.0, 4.0])

result = a / b

print(f"Tensor A is {a}")
print(f"Tensor B is {b}")
print(f"A + B is {result}")


