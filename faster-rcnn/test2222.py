import torch
a = list()
a.append(torch.tensor([1,1]))
a.append(torch.tensor([2,2]))


A=torch.ones(2)
B=2*torch.ones(4)
C=torch.cat([[A,B]],0)
print(C)