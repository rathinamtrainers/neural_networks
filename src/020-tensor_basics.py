import torch

# Scalar (0D tensor)
scalar = torch.tensor(5)
print(f"Scalar: {scalar}")
print(f"Scalar shape: {scalar.shape}")

# Vector (1D tensor)
vector = torch.tensor([1, 2, 3])
print(f"vector: {vector}")
print(f"vector shape: {vector.shape}")

# Matrix (2D tensor)
matrix = torch.tensor([[1,2], [3,4]])
print(f"matrix: {matrix}")
print(f"matrix shape: {matrix.shape}")

# 3D tensor
tensor3d = torch.randn(2, 3, 4)
print(f"tensor3d: {tensor3d}")
print(f"tensor3d shape: {tensor3d.shape}")

# Zero filled tensors
zero_tensor = torch.zeros(2, 3)
print(f"zero_tensor: {zero_tensor}")
print(f"zero_tensor shape: {zero_tensor.shape}")

# One filled tensors
one_tensor = torch.ones(2, 3)
print(f"one_tensor: {one_tensor}")
print(f"one_tensor shape: {one_tensor.shape}")

# Random filled tensors
random_tensor = torch.randn(2, 3)
print(f"random_tensor: {random_tensor}")
print(f"random_tensor shape: {random_tensor.shape}")

# Convert numpy to tensor
import numpy as np
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"tensor_from_numpy: {tensor_from_numpy}")
print(f"tensor_from_numpy shape: {tensor_from_numpy.shape}")

# Convert tensor to numpy
tensor_to_numpy = tensor_from_numpy.numpy()
print(f"tensor_to_numpy: {tensor_to_numpy}")
print(f"tensor_to_numpy shape: {tensor_to_numpy.shape}")

# tensor metadata
print(f"tensor: {tensor3d}")
print(f"tensor shape: {tensor3d.shape}")
print(f"tensor dtype: {tensor3d.dtype}")
print(f"tensor device: {tensor3d.device}")

# Element-wise operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(f"Addition: {a + b}")
print(f"Multiplication: {a * b}")

# Matrix multiplication
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print(f"Matrix multiplication: {torch.matmul(a, b)}")
print(f"Element wise Multiplication in Matrix: {a * b}")

# Reshape
t = torch.arange(0, 12)
print(f"Original: {t}")
reshaped = t.view(3, 4)
print(f"Reshaped: {reshaped}")
reshaped2 = t.reshape(3, 4)
print(f"Reshaped2: {reshaped2}")

# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Move tensor to GPU
tensor_on_gpu = tensor3d.to(device)
print(f"tensor_on_gpu: {tensor_on_gpu}")
print(f"tensor_on_gpu device: {tensor_on_gpu.device}")