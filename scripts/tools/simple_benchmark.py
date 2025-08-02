import torch
import time

# ------------------------
# Configurable parameters
# ------------------------
shape = (10000, 512)   # Size of tensor to transfer (simulate obs/actions)
repeat = 100           # Number of test repetitions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

cpu_tensor = torch.randn(shape, dtype=torch.float32, pin_memory=True)
gpu_tensor = torch.empty(shape, dtype=torch.float32, device=device)

# Warm-up
for _ in range(10):
    _ = cpu_tensor.to(device, non_blocking=True)
    _ = gpu_tensor.to("cpu", non_blocking=True)

torch.cuda.synchronize()

# ---------------------
# Measure CPU -> GPU
# ---------------------
start = time.perf_counter()
for _ in range(repeat):
    x = cpu_tensor.to(device, non_blocking=True)
torch.cuda.synchronize()
end = time.perf_counter()
time_cpu2gpu = (end - start) / repeat * 1000  # ms

# ---------------------
# Measure GPU -> CPU
# ---------------------
start = time.perf_counter()
for _ in range(repeat):
    y = gpu_tensor.to("cpu", non_blocking=True)
torch.cuda.synchronize()
end = time.perf_counter()
time_gpu2cpu = (end - start) / repeat * 1000  # ms

print(f"\n===== Data Transfer Benchmark =====")
print(f"CPU -> GPU avg time: {time_cpu2gpu:.4f} ms")
print(f"GPU -> CPU avg time: {time_gpu2cpu:.4f} ms")
print(f"Tensor shape: {shape}, repeats: {repeat}")

