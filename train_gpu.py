# train_ddp.py
import argparse
import os
import yaml
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from ema_pytorch import EMA

from model.FVP import FVP
from model.vision_3d.pointnet_extractor import PointNetEncoderXYZ

from dataset.metaworld_dataset import MetaworldDataset
from dataset.realdex_dataset import RealDexDataset

# 从你的 utils 导入工具函数/类
from utils import Config, get_optimizer, init_seeds, reduce_tensor, DataLoaderDDP, DataLoaderNonDDP, print0

# ---------- Distributed helpers ----------

def is_distributed():
    return ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1)

def setup_distributed(local_rank):
    if is_distributed():
        # torchrun 已设置 MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank_env = int(os.environ.get("LOCAL_RANK", local_rank))
        torch.cuda.set_device(local_rank_env)
        device = torch.device(f"cuda:{local_rank_env}")
        return device, world_size, rank
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return device, 1, 0

def cleanup_distributed():
    if is_distributed() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

# ---------- Main training ----------

def train(opt):
    
    local_rank, use_amp = opt.local_rank, opt.use_amp

    # read yaml config
    with open(opt.config, 'r') as f:
        cfg = yaml.full_load(f)
    print0("yaml cfg loaded:", cfg)
    opt = Config(cfg)
    
    opt.local_rank = local_rank
    opt.use_amp = use_amp

    # init distributed + device
    device, world_size, rank = setup_distributed(opt.local_rank)
    print0(f"rank {rank} world_size {world_size} device {device}")

    # seed; pass rank to init_seeds so different processes get different seeds
    init_seeds(no=rank)

    model_dir = os.path.join(opt.save_dir, "ckpts")
    vis_dir = os.path.join(opt.save_dir, "visual")
    tsbd_dir = os.path.join(opt.save_dir, "tensorboard")

    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    # Build model and send to device
    viz_encoder = PointNetEncoderXYZ()
    if opt.resume is not None:
        ckpt = torch.load(opt.resume, map_location='cpu')
        state_dict = ckpt['MODEL']
        viz_encoder.load_state_dict(state_dict, strict=True)
    fvp = FVP(encoder=viz_encoder, **opt.diffusion, device=device)
    fvp.to(device)

    # Wrap with DistributedDataParallel if multi-GPU
    if world_size > 1:
        # DDP expects model on current device, wrap and set device_ids
        local_rank_env = int(os.environ.get("LOCAL_RANK", opt.local_rank))
        fvp = torch.nn.parallel.DistributedDataParallel(fvp, device_ids=[local_rank_env], output_device=local_rank_env)

    # EMA & Tensorboard only on rank 0
    ema = None
    writer = None
    if rank == 0:
        model_for_ema = fvp.module if hasattr(fvp, "module") else fvp
        ema = EMA(model_for_ema, beta=opt.ema, update_after_step=0, update_every=1)
        ema.to(device)
        ema.eval()
        writer = SummaryWriter(log_dir=tsbd_dir)

    # Dataset
    train_dataset = RealDexDataset(zarr_path=opt.dataset,
                                     horizon=2,
                                     pad_before=1,
                                     pad_after=0,
                                     seed=42,
                                     val_ratio=0.0,
                                     max_train_episodes=90)
    normalizer = train_dataset.get_normalizer()
    # set normalizer on underlying model (not DDP wrapper)
    model_for_norm = fvp.module if hasattr(fvp, "module") else fvp
    model_for_norm.set_normalizer(normalizer)

    per_proc_batch = max(1, int(opt.batch_size // world_size))
    if rank == 0 and opt.batch_size % world_size != 0:
        print0(f"Warning: opt.batch_size ({opt.batch_size}) not divisible by world_size ({world_size}). "
               f"Using per-process batch {per_proc_batch} (floor division).")

    # DataLoader: use utils.DataLoaderDDP when distributed, else DataLoaderNonDDP
    if world_size > 1:
        train_loader, sampler = DataLoaderDDP(train_dataset, batch_size=per_proc_batch, shuffle=True)
    else:
        train_loader = DataLoaderNonDDP(train_dataset, batch_size=per_proc_batch, shuffle=True)
        sampler = None

    # Optimizer / scaler
    lr = opt.lrate
    optim = get_optimizer([{'params': fvp.parameters(), 'lr': lr * opt.lrate_ratio}], opt, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp if hasattr(opt, "use_amp") else False)

    # optionally load checkpoint
    if opt.load_epoch != -1:
        target = os.path.join(model_dir, f"model_{opt.load_epoch}.pth")
        if rank == 0:
            print0("loading model at", target)
        map_location = {'cuda:%d' % 0: str(device)} if torch.cuda.is_available() else device
        checkpoint = torch.load(target, map_location=map_location)
        model_state = checkpoint.get('MODEL', checkpoint)
        model_to_load = fvp.module if hasattr(fvp, "module") else fvp
        try:
            model_to_load.load_state_dict(model_state)
        except Exception as e:
            print0("Warning: loading MODEL failed, trying direct load. Error:", e)
            model_to_load.load_state_dict(checkpoint)
        if rank == 0 and 'EMA' in checkpoint and ema is not None:
            ema.load_state_dict(checkpoint['EMA'])
        if 'opt' in checkpoint:
            optim.load_state_dict(checkpoint['opt'])

    # training loop
    for ep in range(opt.load_epoch + 1, opt.n_epoch):
        # set epoch for distributed sampler for shuffling
        if sampler is not None:
            sampler.set_epoch(ep)

        # learning rate schedule (warmup)
        optim.param_groups[0]['lr'] = lr * min((ep + 1.0) / opt.warm_epoch, 1.0)
        optim.param_groups[0]['lr'] = optim.param_groups[0]['lr'] * opt.lrate_ratio

        fvp.train()
        if rank == 0:
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader

        loss_ema = None

        for source in pbar:
            optim.zero_grad()
            # move to device
            pointcloud = source['obs']
            pointcloud['point_cloud'] = pointcloud['point_cloud'].to(device, non_blocking=True)
            pointcloud['agent_pos'] = pointcloud['agent_pos'].to(device, non_blocking=True)

            # forward/backward with optional AMP
            with torch.cuda.amp.autocast(enabled=(opt.use_amp if hasattr(opt, "use_amp") else False)):
                loss = fvp(pointcloud, use_amp=False)  # keep your original flag usage

            # Possibly average loss across ranks for logging
            loss_val = loss.detach()
            if world_size > 1:
                # reduce loss to rank 0 for logging
                loss_reduced = reduce_tensor(loss_val)
                loss_num = loss_reduced.item()
            else:
                loss_num = loss_val.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(parameters=fvp.parameters(), max_norm=opt.grad_clip_norm)
            scaler.step(optim)
            scaler.update()

            if rank == 0 and ema is not None:
                ema.update()
                if loss_ema is None:
                    loss_ema = loss_num
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss_num
                pbar.set_description(f"loss: {loss_ema:.4f}")

        # sync processes at epoch end
        if world_size > 1:
            dist.barrier()

        if rank == 0:
            print0("epoch", ep, "loss", loss_ema)
            checkpoint = {
                # save full model state (underlying module if DDP)
                'MODEL': (fvp.module.encoder.state_dict() if hasattr(fvp, "module") else fvp.encoder.state_dict()),
                'opt': optim.state_dict(),
            }
            if ema is not None:
                checkpoint['EMA'] = ema.state_dict()
            save_path = os.path.join(model_dir, f"model_last.pth")
            torch.save(checkpoint, save_path)

    # cleanup
    cleanup_distributed()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--local_rank', default=0, type=int, help='local rank for single-node multi-gpu (torchrun sets LOCAL_RANK env)')
    parser.add_argument("--use_amp", action='store_true', default=False)
    opt = parser.parse_args()
    print0("launch args:", opt)

    # call init_seeds at start (will be called inside train as well)
    init_seeds(no=opt.local_rank)

    train(opt)
