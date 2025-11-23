
import psutil
import GPUtil
import time
import sys

print("Testing psutil.cpu_percent(interval=0.1)...")
start = time.time()
cpu = psutil.cpu_percent(interval=0.1)
print(f"psutil.cpu_percent took {time.time() - start:.4f}s. Result: {cpu}%")

print("Testing GPUtil.getGPUs()...")
start = time.time()
try:
    gpus = GPUtil.getGPUs()
    print(f"GPUtil.getGPUs() took {time.time() - start:.4f}s. Found {len(gpus)} GPUs.")
    for gpu in gpus:
        print(f"GPU: {gpu.name}, Load: {gpu.load}, Memory: {gpu.memoryUsed}/{gpu.memoryTotal}")
except Exception as e:
    print(f"GPUtil failed: {e}")

print("Done.")
