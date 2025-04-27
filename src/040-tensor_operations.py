
import torch

# Arithmetic operations
a = torch.tensor([2.0, 4.0, 6.0])
b = torch.tensor([1.0, 3.0, 5.0])

print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Exponentiation:", a ** 2)


# Matrix Multiplication
mat1 = torch.tensor([[1, 2], [3, 4]])
mat2 = torch.tensor([[2, 0], [1, 2]])

# Matrix multiplication
print("\nMatrix Multiplication:\n", torch.matmul(mat1, mat2))

# Alternatively, using @ operator
print("\nUsing @ operator:\n", mat1 @ mat2)

# Transpose
print("\nTranspose:\n", mat1.T)

# Broadcasting Addition
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([1.0, 2.0])  # 1D tensor

# Broadcasting: y is automatically expanded
print("\nBroadcasted Addition:\n", x + y)


######################################
# Out of place && In place operations
######################################
t = torch.tensor([1, 2, 3])
print("\nOriginal:", t)

# Out-of-place
t_add = t + 5
print("Out-of-place Add:", t_add)

# In-place
t.add_(5)
print("In-place Add:", t)


############################
# Aggregate Functions
###########################
t = torch.tensor([[1.0, 2.0], [13.0, 4.0]])

print("\nSum:", t.sum())
print("Mean:", t.mean())
print("Max:", t.max())
print("Min:", t.min())
print("Argmax (Index of Max):", t.argmax())


####################
# Flattening
t = torch.arange(0, 12)
print("\nOriginal:", t)

reshaped = t.view(3, 4)
print("Reshaped to (3x4):\n", reshaped)

# Flatten
flattened = reshaped.view(-1)
print("Flattened:", flattened)


#######################
# Indexing & Slicing
#######################
t = torch.tensor([[10, 20, 30], [40, 50, 60]])

print("\nOriginal:\n", t)
print("First row:", t[0])
print("Element at (1,2):", t[1, 2])
print("All rows, first column:", t[:, 0])

#############
# Cloning
a = torch.tensor([1, 2, 3])
b = a.clone()
b[0] = 999

print("\nOriginal A:", a)
print("Cloned and Modified B:", b)




