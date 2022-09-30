import torch

a = torch.Tensor(5, 3, 12, 12)
b = torch.sum(a, dim=1, keepdim=True)
c = torch.sum(a, dim=1, keepdim=False)
print(b.shape)
print(c.shape)
