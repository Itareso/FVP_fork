import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
"""
class PointNetEncoder(nn.Module):
    def __init__(self, zdim=64, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.LayerNorm(128)
        self.bn2 = nn.LayerNorm(128)
        self.bn3 = nn.LayerNorm(256)
        self.bn4 = nn.LayerNorm(512)
        
        #self.conv5 = nn.Conv1d(512, 128, 1)
        #self.conv6 = nn.Conv1d(512, 512, 1)
        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(128, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.LayerNorm(256)
        self.fc_bn2_m = nn.LayerNorm(128)
        self.num_layers = 100
        self.conv_layers = nn.ModuleList([ nn.Conv1d(input_dim, 128, 1)])
        self.layernorm_layers = nn.ModuleList([nn.LayerNorm(128)])
        for i in range(1, self.num_layers):
            self.conv_layers.append(nn.Conv1d(128, 128, 1))
            self.layernorm_layers.append(nn.LayerNorm(128))

        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.transpose(1, 2)
        for conv_layer, layernorm_layer in zip(self.conv_layers, self.layernorm_layers):
            x = conv_layer(x).transpose(1, 2)
            x = layernorm_layer(x)
            x = self.relu(x).transpose(1, 2)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
       

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m
"""
class PointNetEncoder(nn.Module):
    def __init__(self, zdim=64, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
    
        self.bn1 = nn.LayerNorm(128)
        self.bn2 = nn.LayerNorm(128)
        self.bn3 = nn.LayerNorm(256)
        self.bn4 = nn.LayerNorm(512)
        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.LayerNorm(256)
        self.fc_bn2_m = nn.LayerNorm(128)

        
             
        #source =  source[:,sampled_indices]

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.bn2(self.conv2(x).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.bn3(self.conv3(x).transpose(1, 2)).transpose(1, 2))
        x = self.bn4(self.conv4(x).transpose(1, 2))
        m = torch.max(x, 1, keepdim=True)[0]
        m = m.view(-1, 512)
        m = F.relu(self.fc_bn1_m(self.fc1_m(m)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        #x = x[:,self.sampled_indices]

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return  x, m    
if __name__ == "__main__":
    x = torch.rand(10, 1024, 3).cuda(3)
    model = PointNetEncoder().cuda(3)
    y = model(x)
    print(y.shape)
