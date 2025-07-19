from functools import partial
import os
import math
import inspect
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from .attention import MultiHeadAttention
from .block import TimeEmbedding
from .point_cloud_model import PointCloudModel
from model.common.normalizer import LinearNormalizer
def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


def linear_beta_schedule(timesteps, beta1, beta2):
    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    return torch.linspace(beta1, beta2, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps # dtype = torch.float64
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def inverted_cosine_beta_schedule(timesteps, s = 0.008):

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps # dtype = torch.float64
    alphas_cumprod = (2 * (1 + s) / math.pi) * torch.arccos(torch.sqrt(t)) - s
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def schedules(betas, T, device, type='DDPM'):
    if betas == 'inverted':
        schedule_fn = inverted_cosine_beta_schedule
    elif betas == 'cosine':
        schedule_fn = cosine_beta_schedule
    else:
        beta1, beta2 = betas
        beta1 = 1e-5
        beta2 = 8e-3
        schedule_fn = partial(linear_beta_schedule, beta1=beta1, beta2=beta2)

    if type == 'DDPM':
        beta_t = torch.cat([torch.tensor([0.0]), schedule_fn(T)])
    elif type == 'DDIM':
        beta_t = schedule_fn(T + 1)
    else:
        raise NotImplementedError()
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    ma_over_sqrtmab = (1 - alpha_t) / sqrtmab

    dic = {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "ma_over_sqrtmab": ma_over_sqrtmab,
    }
    return {key: dic[key].to(device) for key in dic}


class FVP(nn.Module):
    def __init__(self, encoder, betas, n_T, drop_prob, model_input,model_output,model_embed,
                 embed_predict_in,embed_predict_out,embed_in,embed_out, device):
        super(FVP, self).__init__()
        self.encoder = encoder.to(device)
        # self.decoder = decoder.to(device)
        # if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        #     params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6
        #     print(f"encoder # params: {params:.1f}")
        #     params = sum(p.numel() for p in decoder.parameters() if p.requires_grad) / 1e6
        #     print(f"decoder # params: {params:.1f}")

        self.device = device
        self.ddpm_sche = schedules(betas, n_T, device, 'DDPM')
        self.ddim_sche = schedules(betas, n_T, device, 'DDIM')
        self.n_T = n_T
        self.drop_prob = drop_prob
        self.loss =  nn.MSELoss()

        self.point_cloud_model = PointCloudModel(
            model_type='pvcnn',
            embed_dim=model_embed,
            in_channels=model_input,
            out_channels=model_output,
        )
        self.embed_predict = nn.Sequential(nn.Linear(embed_predict_in,embed_predict_out),
                                           nn.ReLU())
        self.embed = nn.Sequential(nn.Linear(embed_in,embed_out),
                                   nn.ReLU())

                                          
        scheduler_kwargs = {}
        scheduler_kwargs.update(dict(beta_start=1e-5, beta_end=8e-3, beta_schedule='linear'))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
        }
        
        self.scheduler = self.schedulers_map['ddpm']  
        self.normalizer = LinearNormalizer()
    def perturb(self, x, t=None):
        
        noise = torch.randn_like(x)
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (x.shape[0],), 
            device=self.device, dtype=torch.long)

        x_t = self.scheduler.add_noise(x, noise, timestep)

        return x_t, timestep, noise
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def forward(self, pointcloud, use_amp=False):

        self.normalizer.fit(pointcloud)
        nobs = self.normalizer.normalize(pointcloud)
        

        pointcloud = nobs['point_cloud']
        
        x_past = pointcloud[:,0,:,0:3]
        x_target  = pointcloud[:,1,:,0:3] 
        B, N, _ = x_target.shape
       
        x_noised, t, noise = self.perturb(x_target , t=None)
        with autocast(enabled=use_amp):
            z_past_m,z_past_x = self.encoder(x_past)
            z_past_x = z_past_x.unsqueeze(1).expand(-1,N,-1)
            z_past_m = self.embed(z_past_m)
            z_past_x = self.embed_predict(z_past_x)
            z = torch.cat((x_noised,z_past_m,z_past_x), dim=-1)
            re_noise = self.point_cloud_model(z , t)
            loss = self.loss(noise,re_noise)
            return loss
        
    def encode(self, x, norm=False, use_amp=False):
        with autocast(enabled=use_amp):
            z = self.encoder(x)
        if norm:
            z = torch.nn.functional.normalize(z)
        return z
    
    def ddim_sample(self, n_sample, size,past,future, steps=100, eta=0.0, guide_w=0.3, notqdm=False, use_amp=False):
        scheduler = self.scheduler
        x_t = torch.randn(n_sample, *size).to(self.device)
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(self.n_T, **extra_set_kwargs)
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}
        progress_bar = tqdm(scheduler.timesteps.to(self.device), desc=f'Sampling ({x_t.shape})', disable=False)
        z_past,_= self.encoder(future)  
        z_past = self.embed(z_past)
        for i, t in enumerate(progress_bar):
            
           
            z = torch.cat((z_past, x_t), dim=-1)
            noise_pred = self.point_cloud_model(z, t.reshape(1).expand(z.shape[0]))
     
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
       
        return x_t
    
    def pred_eps_(self, x, t, model_args, guide_w, alpha, use_amp, clip_x=True):
        def pred_cfg_eps_double_batch():
            x_double = x.repeat(2, 1, 1)
            t_double = t.repeat(2)

            with autocast(enabled=use_amp):
                eps = self.point_cloud_model(x_double, t_double).float()
            n_sample = eps.shape[0] // 2
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            assert eps1.shape == eps2.shape
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            return eps

        def pred_eps_from_x0(x0):
            return (x[:,:,3:6]- x0 * alpha.sqrt()) / (1 - alpha).sqrt()

        def pred_x0_from_eps(eps):
            return (x [:,:,3:6]- (1 - alpha).sqrt() * eps) / alpha.sqrt()

        eps = pred_cfg_eps_double_batch()
        denoised = pred_x0_from_eps(eps)

        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
            eps = pred_eps_from_x0(denoised)
        return eps, denoised


    

