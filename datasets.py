import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import h5py
import numpy as np
from torchvision import datasets
import numpy as np
#import pytorch3d.ops as torch3d_ops
import torch
import os

# os.environ['LOCAL_RANK']
# print(LOCAL_RANK)
"""
def point_cloud_sampling(point_cloud:np.ndarray, num_points:int, method:str='fps'):
    
    if num_points == 'all': # use all points
        return point_cloud

    if point_cloud.shape[0] <= num_points:
        # cprint(f"warning: point cloud has {point_cloud.shape[0]} points, but we want to sample {num_points} points", 'yellow')
        # pad with zeros
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], point_cloud_dim))], axis=0)
        return point_cloud

    if method == 'uniform':
        # uniform sampling
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[sampled_indices]
    elif method == 'fps':
        # fast point cloud sampling using torch3d
        point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points = torch.tensor([num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        raise NotImplementedError(f"point cloud sampling method {method} not implemented")

    return point_cloud
"""
class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        if not root.endswith("tiny-imagenet-200"):
            root = os.path.join(root, "tiny-imagenet-200")
        self.train_dir = os.path.join(root, "train")
        self.val_dir = os.path.join(root, "val")
        self.transform = transform
        if train:
            self._scan_train()
        else:
            self._scan_val()

    def _scan_train(self):
        classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        classes = sorted(classes)
        assert len(classes) == 200

        self.data = []
        for idx, name in enumerate(classes):
            this_dir = os.path.join(self.train_dir, name)
            for root, _, files in sorted(os.walk(this_dir)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        item = (path, idx)
                        self.data.append(item)
        self.labels_dict = {i: classes[i] for i in range(len(classes))}

    def _scan_val(self):
        self.file_to_class = {}
        classes = set()
        with open(os.path.join(self.val_dir, "val_annotations.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            words = line.split("\t")
            self.file_to_class[words[0]] = words[1]
            classes.add(words[1])
        classes = sorted(list(classes))
        assert len(classes) == 200

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.data = []
        this_dir = os.path.join(self.val_dir, "images")
        for root, _, files in sorted(os.walk(this_dir)):
            for fname in sorted(files):
                if fname.endswith(".JPEG"):
                    path = os.path.join(root, fname)
                    idx = class_to_idx[self.file_to_class[fname]]
                    item = (path, idx)
                    self.data.append(item)
        self.labels_dict = {i: classes[i] for i in range(len(classes))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label



########################################################   hoi4d dataset

class SegDataset(Dataset):
    def __init__(self, root=None, train=True):
        super(SegDataset, self).__init__()
        self.train = train
        self.pcd = []
        self.center = []
        self.label = []
         
        if self.train:
            #for filename in ['train1.h5','train2.h5','train3.h5','train4.h5']:
            for filename in ['train1.h5']:
                print(filename)
                with h5py.File(root+'/'+filename,'r') as f:
                    self.pcd.append(np.array(f['pcd']))
                    self.center.append(np.array(f['center']))
                    self.label.append(np.array(f['label']))
        else:
            for filename in ['test1.h5', 'test2.h5']:
                print(filename)
                with h5py.File(root+'/'+filename,'r') as f:
                    self.pcd.append(np.array(f['pcd']))
                    self.center.append(np.array(f['center']))
                    self.label.append(np.array(f['label']))
        self.pcd = np.concatenate(self.pcd, axis=0)
        self.center = np.concatenate(self.center,axis=0)
        self.label = np.concatenate(self.label,axis=0)

    def __len__(self):
        return len(self.pcd)

    def augment(self, pc, center):
      
        flip = np.random.uniform(0, 1) > 0.5
        if flip:
            pc = (pc - center)
            pc[:, 0] *= -1
            pc += center
        else:
            pc = pc - center
            jittered_data = np.clip(0.01 * np.random.randn(150,2048,3), -1*0.05, 0.05)
            jittered_data += pc
            pc = pc + center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center
        pc = pc /20
        return pc

    def __getitem__(self, index):
        pc = self.pcd[index]
        center_0 = self.center[index][0]
        label = self.label[index]
        if self.train:
            pc = self.augment(pc, center_0)
        return pc.astype(np.float32), label.astype(np.int64)


########################################################

class SourceTargetDataset(Dataset):
    def __init__(self, source):
        self.source = source
        

    def __getitem__(self, index):
        video, _= self.source.__getitem__(index)
        index1 = np.random.choice(120)
        image_1 = video[index1]
        image_2 = video[index1+1] 
        image_3= video[index1-1] 
        #image_1 = point_cloud_sampling(image_1,512,'uniform')
        #image_2 = point_cloud_sampling(image_2,512,'uniform')
        #assert np.all(label_1 == label_2)
        return image_1, image_2, image_3

    def __len__(self):
        l1 = self.source.__len__()
        #assert l1 == l2
        return l1


class AddGaussianNoise():
    def __init__(self, sigma=0.10):
        self.sigma = sigma

    def __call__(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        dtype = tensor.dtype

        tensor = tensor.float()
        out = tensor + self.sigma * torch.randn_like(tensor)

        if out.dtype != dtype:
            out = out.to(dtype)
        return out


def get_dataset(name='cifar10', root='data'):
    if name == 'cifar10':
        data_norm = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        NUM_CLASSES = 10
        DATASET = CIFAR10
        RES = 32
    elif name == 'cifar100':
        data_norm = transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2009, 0.1984, 0.2023])
        NUM_CLASSES = 100
        DATASET = CIFAR100
        RES = 32
    elif name == 'tiny':
        data_norm = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        NUM_CLASSES = 200
        DATASET = TinyImageNet
        RES = 64
    #changed by hsb
    elif name=='hoi4d':
        data_norm = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        NUM_CLASSES = 200
        DATASET = SegDataset
        RES = 32
    else:
        raise NotImplementedError

    # for resnet encoder, at the training
    source_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(RES),
            transforms.RandomHorizontalFlip(),
        ], p=0.95),
        transforms.RandomApply([
            transforms.RandAugment(),
        ], p=0.65),
        transforms.ToTensor(),
        data_norm,
        AddGaussianNoise(),
    ])
    # for unet decoder, at the training
    target_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(RES),
            transforms.RandomHorizontalFlip(),
        ], p=0.95),
        transforms.RandomApply([
            transforms.RandAugment(),
        ], p=0.65),
        transforms.ToTensor(),
    ])
    # for resnet encoder, for evaluation
    downstream_transform_train = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(RES),
            transforms.RandomHorizontalFlip(),
        ], p=0.65),
        transforms.ToTensor(),
        data_norm,
    ])
    downstream_transform_test = transforms.Compose([
        transforms.ToTensor(),
        data_norm,
    ])

    if name=='hoi4d':
        train_source = DATASET(root=root, train=True)
    else:
        train_source = DATASET(root=root, train=True, transform=source_transform)
        train_target = DATASET(root=root, train=True, transform=target_transform)
        down_train = DATASET(root=root, train=True, transform=downstream_transform_train)
        down_test = DATASET(root=root, train=False, transform=downstream_transform_test)
    
    train_source_target = SourceTargetDataset(train_source)


    return  train_source_target


import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm


synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNetCore(Dataset):

    GRAVITATIONAL_AXIS = 1
    
    def __init__(self, path, cates, split, scale_mode, transform=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            cates = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds = []
        self.stats = None

        self.get_statistics()
        self.load()

    def get_statistics(self):

        basename = os.path.basename(self.path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        with h5py.File(self.path, 'r') as f:
            pointclouds = []
            for synsetid in self.cate_synsetids:
                for split in ('train', 'val', 'test'):
                    pointclouds.append(torch.from_numpy(f[synsetid][split][...]))

        all_points = torch.cat(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def load(self):

        def _enumerate_pointclouds(f):
            for synsetid in self.cate_synsetids:
                cate_name = synsetid_to_cate[synsetid]
                for j, pc in enumerate(f[synsetid][self.split]):
                    yield torch.from_numpy(pc), j, cate_name
        
        with h5py.File(self.path, mode='r') as f:
            for pc, pc_id, cate_name in _enumerate_pointclouds(f):

                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats['std'].reshape(1, 1)
                elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])

                pc = (pc - shift) / scale

                self.pointclouds.append({
                    'pointcloud': pc,
                    'cate': cate_name,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data



if __name__ == '__main__':
    train = get_dataset(name='hoi4d', root='/localdata/houchengkai/hoi4d/actionseg')
    a, b = train.__getitem__(1)
    print(a.shape)
    print(np.max(a, axis=0)) 
    print(np.min(a, axis=0))
