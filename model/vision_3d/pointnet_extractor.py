import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from model.vision_3d.positional_encoding import PositionalEncoding
#from positional_encoding import PositionalEncoding
from model.vision_3d.pointnet2_new_encoder import PointNet2EncoderXYZ
from model.vision_3d.pointnext_encoder import PointNextEncoderXYZ
from model.vision_3d.pointnet_new import PointNetEncoder
try:
    from model.vision_3d.voxelcnn_encoder import VoxelCNN
except:
    print("voxel cnn not found. pass")
from model.vision_3d.pointtransformer_encoder import Backbone as PT_Backbone
#from pointtransformer_encoder import Backbone as PT_Backbone
from matplotlib import pyplot as plt

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, simnorm_dim=8):
		super().__init__()
		self.dim = simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"



class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=64,
                 use_layernorm: bool=True,
                 final_norm: str='layernorm',
                 #use_layernorm: bool=False,
                 #final_norm: str='none',
                 use_positional_encoding: bool=False,
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_in=in_channels, include_input=True)
            in_channels = self.positional_encoding.d_out
            cprint(f"[PointNetEncoderXYZ] use positional encoding, in_channels: {in_channels}", "green")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'simnorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                SimNorm(),
            )
        elif final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")


    def forward(self, x):
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        x = self.mlp(x)
        m = torch.max(x, 1)[0]
        m = self.final_projection(m)
        return x,m


class PointTransformer(nn.Module):
    """Encoder for pointtransformer
    """
    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=64,
                 normal_channel: bool=False,
                 final_norm: str='none',
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            out_channels (int): feature size of output
            normal_channel (bool): whether to use RGB. Defaults to False.
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normal_channel = normal_channel

        if self.normal_channel:
            assert in_channels == 6, cprint(f"PointTransformer only supports 6 channels, but got {in_channels}", "red")
        else:
            assert in_channels == 3, cprint(f"PointTransformer only supports 3 channels, but got {in_channels}", "red")

        self.transformer = PT_Backbone(512,4,16,128,self.in_channels)#number of points, number of blocks, number of nearest neighbors(knn), transformer hidden dimension, dimension of points(default=3)
        if final_norm == 'simnorm':
            self.final_projection = nn.Sequential(
                nn.Linear(512, out_channels),
                SimNorm(),
            )
        elif final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(512, out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(512, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x):
        features, _ = self.transformer(x)
        x = torch.max(self.final_projection(features),dim=1)[0]
        return features,x
if __name__ == "__main__":
    x = torch.rand(10, 1024, 3).cuda(3)
    model = PointNetEncoderXYZ().cuda(3)
    f,y = model(x)
    print(f.shape)
    print(y.shape)


