import torch

t = torch.randn(1,2)

t.to('cuda')

print(t)