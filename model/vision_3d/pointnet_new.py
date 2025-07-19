import torch
import torch.nn.functional as F
from torch import nn

"""

class PointNetEncoder(nn.Module):
    def __init__(self, zdim=64, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv5 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.LayerNorm(32)
        self.bn2 = nn.LayerNorm(64)
        self.bn3 = nn.LayerNorm(256)
        self.bn4 = nn.LayerNorm(512)
        self.bn5 = nn.LayerNorm(128)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.LayerNorm(256)
        self.fc_bn2_m = nn.LayerNorm(128)



    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.bn2(self.conv2(x).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.bn5(self.conv5(x).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.bn3(self.conv3(x).transpose(1, 2)).transpose(1, 2))
        x = self.bn4(self.conv4(x).transpose(1, 2)).transpose(1, 2)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
       

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m
 """  
import torch
import torch.nn.functional as F
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, zdim=64, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Linear(input_dim, 128)
        self.conv2 = nn.Linear(128, 128)
        self.conv3 = nn.Linear(128, 256)
        self.conv4 = nn.Linear(256, 512)
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

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.LayerNorm(256)
        self.fc_bn2_v = nn.LayerNorm(128)

    def forward(self, x):
        #x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
      

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m
if __name__ == "__main__":
    x = torch.rand(10, 512, 3).cuda(3)
    model = PointNetEncoder().cuda(3)
    y = model(x)
    print(y.shape)
