import torch
import torch.nn.functional as F
a = torch.randn(10,512,256)
b = F.normalize(a, dim=1)
print(b.size())
#b = torch.max(b,dim=2)[0].unsqueeze(1)
#print(b.size())z
b = b*a
print(b.size())
b= torch.sum(b,dim=1)/512
print(b.size())



net = torch.load("/data/taskmyanythin/3D-Diffusion-Policy/3D-Diffusion-Policy/soda/dp3/ckpts/model_last.pth")
for key in net:
    print(key)
#for key, value in net["state_dict"].items():
#     print(key,value.size(),sep="  ")