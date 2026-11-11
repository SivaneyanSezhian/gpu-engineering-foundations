import torch

A = torch.rand(size=(10, 10))
B = torch.rand(size=(10, 10))
C = A @ B

print(C)
