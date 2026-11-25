import sys, torch

print("Python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available())