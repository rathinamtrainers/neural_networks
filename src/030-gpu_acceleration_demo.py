import torch
import time

size = 1000
a_cpu = torch.rand(size, size)
b_cpu = torch.rand(size, size)

# CPU timing
start = time.time()
for _ in range(100):
    result_cpu = torch.matmul(a_cpu, b_cpu)
end = time.time()
print(f"CPU time: {end - start}")

# GPU timing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a_gpu = a_cpu.to(device)
b_gpu = b_cpu.to(device)

# Warm up GPU
_ = torch.matmul(a_gpu, b_gpu)

start = time.time()
for _ in range(100):
    result_gpu = torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize()    # Wait for GPU to complete all operations
end = time.time()
print(f"GPU time: {end - start}")

