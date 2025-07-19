import argparse
import os
from functools import partial

import torch
import torch.distributed as dist
import  numpy as np
import yaml
from metric import KNN, LinearProbe
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ema_pytorch import EMA
from model.FVP import FVP
from model.pointnet import PointNetEncoder
from model.decoder import UNet_decoder
from model.encoder import Network
from utils import Config, get_optimizer, init_seeds, reduce_tensor, DataLoaderDDP,DataLoaderNonDDP, print0
from model.vision_3d.pointnet_extractor import PointNetEncoderXYZ


from datasets import get_dataset,ShapeNetCore
from dataset.realdex_dataset import RealDexDataset
from dataset.metaworld_dataset import MetaworldDataset



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(opt):

    yaml_path = opt.config
    local_rank = opt.local_rank
    use_amp = opt.use_amp

    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print0(opt)
    opt = Config(opt)
    model_dir = os.path.join(opt.save_dir, "ckpts")
    vis_dir = os.path.join(opt.save_dir, "visual")
    tsbd_dir = os.path.join(opt.save_dir, "tensorboard")
    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    device = "cuda:0" #% local_rank
    

    fvp = FVP(encoder= PointNetEncoderXYZ(),
                **opt.diffusion,
                device=device)
    fvp.to(device)




    if local_rank == 0:
        ema = EMA(fvp, beta=opt.ema, update_after_step=0, update_every=1)
        ema.to(device)
        ema.eval()
        writer = SummaryWriter(log_dir=tsbd_dir)

    train = MetaworldDataset(zarr_path = opt.dataset,
            horizon=2,
            pad_before=1,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=90)
    

    normalizer = train.get_normalizer()
    
    train_loader = DataLoaderNonDDP(train,
                                          batch_size=opt.batch_size,
                                          shuffle=True)
    

    lr = opt.lrate
  
    fvp.set_normalizer(normalizer)
    optim = get_optimizer([{'params': fvp.parameters(), 'lr': lr * opt.lrate_ratio},
                           ], opt, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if opt.load_epoch != -1:
        target = os.path.join(model_dir, f"model_{opt.load_epoch}.pth")
        print0("loading model at", target)
        checkpoint = torch.load(target, map_location=device)
        fvp.load_state_dict(checkpoint['MODEL'])
        if local_rank == 0:
            ema.load_state_dict(checkpoint['EMA'])
        optim.load_state_dict(checkpoint['opt'])
    fvp.to(device)
    for ep in range(opt.load_epoch + 1, opt.n_epoch):

        optim.param_groups[0]['lr'] = lr * min((ep + 1.0) / opt.warm_epoch, 1.0) # warmup
        optim.param_groups[0]['lr'] = optim.param_groups[0]['lr'] * opt.lrate_ratio
      
        fvp.train()
        if local_rank == 0:
            enc_lr = optim.param_groups[0]['lr']
            dec_lr = optim.param_groups[0]['lr']
            print(f'epoch {ep}, lr {enc_lr:f} & {dec_lr:f}')
            loss_ema = None
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
        i = 0 
        for source in pbar:
            optim.zero_grad()
        
            pointcloud = source['obs']
            pointcloud['point_cloud'] = pointcloud['point_cloud'].to(device)
            pointcloud['agent_pos'] = pointcloud['agent_pos'].to(device)
            
            loss=fvp(pointcloud, use_amp=False)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(parameters=fvp.parameters(), max_norm=opt.grad_clip_norm)
            scaler.step(optim)
            scaler.update()


            if local_rank == 0:
                ema.update()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")
            i = i+1
        print("epoch", ep, "loss", loss_ema)
        checkpoint = {
               'MODEL': fvp.encoder.state_dict(),}
        save_path = os.path.join(model_dir, f"model_last.pth")
        torch.save(checkpoint, save_path)
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--use_amp", action='store_true', default=False)
    opt = parser.parse_args()
    print0(opt)

    init_seeds(no=opt.local_rank)
    torch.cuda.set_device(opt.local_rank)
    train(opt)