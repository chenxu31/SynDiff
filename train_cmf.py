

import argparse
import torch
import numpy as np
import numpy
import os

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms

import shutil
from skimage.metrics import structural_similarity as SSIM
import platform
import skimage.io
import sys
import pdb

if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_cmf_pt as common_cmf



def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one
#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init[:,[0],:]
    source = x_init[:,[1],:]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
            x_0 = generator(torch.cat((x,source),axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:,[0],:], x, t)
            x = x_new.detach()
        
    return x

#%%
def train_syndiff(device, args):
    gpu = 0
    
    from backbones.discriminator import Discriminator_small, Discriminator_large
    
    from backbones.ncsnpp_generator_adagn import NCSNpp
    
    import backbones.generator_resnet 
    
    
    from utils.EMA import EMA
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    batch_size = args.batch_size
    
    nz = args.nz #latent dimension

    dataset_s = common_cmf.Dataset(args.input_path, "ct", n_slices=1, debug=args.debug)
    dataset_t = common_cmf.Dataset(args.input_path, "mri", n_slices=1, debug=args.debug)

    data_loader_s = torch.utils.data.DataLoader(dataset_s,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               drop_last = True)
    data_loader_t = torch.utils.data.DataLoader(dataset_t,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               drop_last = True)

    val_data_t, val_data_s, _ = common_cmf.load_test_data(args.input_path)

    print('train data size:'+str(min(len(data_loader_s), len(data_loader_t))))
    to_range_0_1 = lambda x: (x + 1.) / 2.

    #networks performing reverse denoising
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)  
    #networks performing translation
    args.num_channels=1
    gen_non_diffusive_1to2 = backbones.generator_resnet.define_G(netG='resnet_6blocks',gpu_ids=[gpu])
    gen_non_diffusive_2to1 = backbones.generator_resnet.define_G(netG='resnet_6blocks',gpu_ids=[gpu])
    
    disc_diffusive_1 = Discriminator_large(nc = 2, ngf = args.ngf, 
                                   t_emb_dim = args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    disc_diffusive_2 = Discriminator_large(nc = 2, ngf = args.ngf, 
                                   t_emb_dim = args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    
    disc_non_diffusive_cycle1 = backbones.generator_resnet.define_D(gpu_ids=[gpu])
    disc_non_diffusive_cycle2 = backbones.generator_resnet.define_D(gpu_ids=[gpu])

    optimizer_disc_diffusive_1 = optim.Adam(disc_diffusive_1.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizer_disc_diffusive_2 = optim.Adam(disc_diffusive_2.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    
    optimizer_gen_diffusive_1 = optim.Adam(gen_diffusive_1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_gen_diffusive_2 = optim.Adam(gen_diffusive_2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    
    optimizer_gen_non_diffusive_1to2 = optim.Adam(gen_non_diffusive_1to2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_gen_non_diffusive_2to1 = optim.Adam(gen_non_diffusive_2to1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))

    optimizer_disc_non_diffusive_cycle1 = optim.Adam(disc_non_diffusive_cycle1.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizer_disc_non_diffusive_cycle2 = optim.Adam(disc_non_diffusive_cycle2.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))    
    
    if args.use_ema:
        optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=args.ema_decay)
        optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=args.ema_decay)
        optimizer_gen_non_diffusive_1to2 = EMA(optimizer_gen_non_diffusive_1to2, ema_decay=args.ema_decay)
        optimizer_gen_non_diffusive_2to1 = EMA(optimizer_gen_non_diffusive_2to1, ema_decay=args.ema_decay)
        
    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_1to2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_1to2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_2to1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_2to1, args.num_epoch, eta_min=1e-5)    
    
    scheduler_disc_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_2, args.num_epoch, eta_min=1e-5)

    scheduler_disc_non_diffusive_cycle1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_non_diffusive_cycle2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle2, args.num_epoch, eta_min=1e-5)
    
    exp = args.exp
    output_path = args.output_path

    exp_path = os.path.join(output_path,exp)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        copy_source(__file__, exp_path)
        shutil.copytree('./backbones', os.path.join(exp_path, 'backbones'))
    
    
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
        gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])
        gen_non_diffusive_1to2.load_state_dict(checkpoint['gen_non_diffusive_1to2_dict'])
        gen_non_diffusive_2to1.load_state_dict(checkpoint['gen_non_diffusive_2to1_dict'])        
        # load G
        
        optimizer_gen_diffusive_1.load_state_dict(checkpoint['optimizer_gen_diffusive_1'])
        scheduler_gen_diffusive_1.load_state_dict(checkpoint['scheduler_gen_diffusive_1'])
        optimizer_gen_diffusive_2.load_state_dict(checkpoint['optimizer_gen_diffusive_2'])
        scheduler_gen_diffusive_2.load_state_dict(checkpoint['scheduler_gen_diffusive_2']) 
        optimizer_gen_non_diffusive_1to2.load_state_dict(checkpoint['optimizer_gen_non_diffusive_1to2'])
        scheduler_gen_non_diffusive_1to2.load_state_dict(checkpoint['scheduler_gen_non_diffusive_1to2'])
        optimizer_gen_non_diffusive_2to1.load_state_dict(checkpoint['optimizer_gen_non_diffusive_2to1'])
        scheduler_gen_non_diffusive_2to1.load_state_dict(checkpoint['scheduler_gen_non_diffusive_2to1'])          
        # load D
        disc_diffusive_1.load_state_dict(checkpoint['disc_diffusive_1_dict'])
        optimizer_disc_diffusive_1.load_state_dict(checkpoint['optimizer_disc_diffusive_1'])
        scheduler_disc_diffusive_1.load_state_dict(checkpoint['scheduler_disc_diffusive_1'])

        disc_diffusive_2.load_state_dict(checkpoint['disc_diffusive_2_dict'])
        optimizer_disc_diffusive_2.load_state_dict(checkpoint['optimizer_disc_diffusive_2'])
        scheduler_disc_diffusive_2.load_state_dict(checkpoint['scheduler_disc_diffusive_2'])   
        # load D_for cycle
        disc_non_diffusive_cycle1.load_state_dict(checkpoint['disc_non_diffusive_cycle1_dict'])
        optimizer_disc_non_diffusive_cycle1.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle1'])
        scheduler_disc_non_diffusive_cycle1.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle1'])

        disc_non_diffusive_cycle2.load_state_dict(checkpoint['disc_non_diffusive_cycle2_dict'])
        optimizer_disc_non_diffusive_cycle2.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle2'])
        scheduler_disc_non_diffusive_cycle2.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle2'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
    best_psnr = 0
    for epoch in range(init_epoch, args.num_epoch+1):
        for iteration, (data1, data2) in enumerate(zip(data_loader_s, data_loader_t)):
            x1 = data1["image"]
            x2 = data2["image"]

            for p in disc_diffusive_1.parameters():  
                p.requires_grad = True  
            for p in disc_diffusive_2.parameters():  
                p.requires_grad = True
            for p in disc_non_diffusive_cycle1.parameters():  
                p.requires_grad = True  
            for p in disc_non_diffusive_cycle2.parameters():  
                p.requires_grad = True          
            
            disc_diffusive_1.zero_grad()
            disc_diffusive_2.zero_grad()
            
            #sample from p(x_0)
            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)
            
            #sample t
            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            #sample x_t and x_tp1
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x1_t.requires_grad = True
            
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
            x2_t.requires_grad = True               
            # train discriminator with real                              
            D1_real = disc_diffusive_1(x1_t, t1, x1_tp1.detach()).view(-1)
            D2_real = disc_diffusive_2(x2_t, t2, x2_tp1.detach()).view(-1)   
            
            errD1_real = F.softplus(-D1_real)
            errD1_real = errD1_real.mean()            
            
            errD2_real = F.softplus(-D2_real)
            errD2_real = errD2_real.mean()   
            errD_real = errD1_real + errD2_real
            errD_real.backward(retain_graph=True)
            
            if args.lazy_reg is None:
                grad1_real = torch.autograd.grad(
                            outputs=D1_real.sum(), inputs=x1_t, create_graph=True
                            )[0]
                grad1_penalty = (
                                grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                grad2_real = torch.autograd.grad(
                            outputs=D2_real.sum(), inputs=x2_t, create_graph=True
                            )[0]
                grad2_penalty = (
                                grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()                
                
                grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad1_real = torch.autograd.grad(
                            outputs=D1_real.sum(), inputs=x1_t, create_graph=True
                            )[0]
                    grad1_penalty = (
                                grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                    grad2_real = torch.autograd.grad(
                            outputs=D2_real.sum(), inputs=x2_t, create_graph=True
                            )[0]
                    grad2_penalty = (
                                grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()                
                
                    grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
                    grad_penalty.backward()
    
            # train with fake
            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)
            
            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)            
            #x_tp1 is concatenated with source contrast and x_0_predict is predicted
            x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(),x2_0_predict),axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(),x1_0_predict),axis=1), t2, latent_z2)
            #sampling q(x_t | x_0_predict, x_t+1)
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:,[0],:], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:,[0],:], x2_tp1, t2)
            #D output for fake sample x_pos_sample
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)       
            
            errD1_fake = F.softplus(output1)
            errD2_fake = F.softplus(output2)
            errD_fake = errD1_fake.mean() + errD2_fake.mean()
            errD_fake.backward()    
            
            errD = errD_real + errD_fake
            # Update D
            optimizer_disc_diffusive_1.step()
            optimizer_disc_diffusive_2.step()  

            #D for cycle part
            disc_non_diffusive_cycle1.zero_grad()
            disc_non_diffusive_cycle2.zero_grad()
            
            #sample from p(x_0)
            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)

            D_cycle1_real = disc_non_diffusive_cycle1(real_data1).view(-1)
            D_cycle2_real = disc_non_diffusive_cycle2(real_data2).view(-1) 
            
            errD_cycle1_real = F.softplus(-D_cycle1_real)
            errD_cycle1_real = errD_cycle1_real.mean()            
            
            errD_cycle2_real = F.softplus(-D_cycle2_real)
            errD_cycle2_real = errD_cycle2_real.mean()   
            errD_cycle_real = errD_cycle1_real + errD_cycle2_real
            errD_cycle_real.backward(retain_graph=True)
            # train with fake
            
            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)

            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1) 
            
            errD_cycle1_fake = F.softplus(D_cycle1_fake)
            errD_cycle1_fake = errD_cycle1_fake.mean()            
            
            errD_cycle2_fake = F.softplus(D_cycle2_fake)
            errD_cycle2_fake = errD_cycle2_fake.mean()   
            errD_cycle_fake = errD_cycle1_fake + errD_cycle2_fake
            errD_cycle_fake.backward()

            errD_cycle = errD_cycle_real + errD_cycle_fake
            # Update D
            optimizer_disc_non_diffusive_cycle1.step()
            optimizer_disc_non_diffusive_cycle2.step() 

            #G part
            for p in disc_diffusive_1.parameters():
                p.requires_grad = False
            for p in disc_diffusive_2.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle1.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle2.parameters():
                p.requires_grad = False                
            gen_diffusive_1.zero_grad()
            gen_diffusive_2.zero_grad()
            gen_non_diffusive_1to2.zero_grad()
            gen_non_diffusive_2to1.zero_grad()   
            
            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            
            #sample x_t and x_tp1            
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)   
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)             
            
            latent_z1 = torch.randn(batch_size, nz,device=device)
            latent_z2 = torch.randn(batch_size, nz,device=device)
            
            #translation networks
            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict_cycle = gen_non_diffusive_1to2(x1_0_predict)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)            
            x1_0_predict_cycle = gen_non_diffusive_2to1(x2_0_predict)   


            #x_tp1 is concatenated with source contrast and x_0_predict is predicted
            x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(),x2_0_predict),axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(),x1_0_predict),axis=1), t2, latent_z2)            
            #sampling q(x_t | x_0_predict, x_t+1)
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:,[0],:], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:,[0],:], x2_tp1, t2)
            #D output for fake sample x_pos_sample
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)  
               
            
            errG1 = F.softplus(-output1)
            errG1 = errG1.mean()

            errG2 = F.softplus(-output2)
            errG2 = errG2.mean()
            
            errG_adv = errG1 + errG2

            #D_cycle output for fake x1_0_predict
            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1) 
            
            errG_cycle_adv1 = F.softplus(-D_cycle1_fake)
            errG_cycle_adv1 = errG_cycle_adv1.mean()            
            
            errG_cycle_adv2 = F.softplus(-D_cycle2_fake)
            errG_cycle_adv2 = errG_cycle_adv2.mean()   
            errG_cycle_adv = errG_cycle_adv1 + errG_cycle_adv2
            
            #L1 loss 
            errG1_L1 = F.l1_loss(x1_0_predict_diff[:,[0],:],real_data1)
            errG2_L1 = F.l1_loss(x2_0_predict_diff[:,[0],:],real_data2)
            errG_L1 = errG1_L1 + errG2_L1 
            
            #cycle loss
            errG1_cycle=F.l1_loss(x1_0_predict_cycle,real_data1)
            errG2_cycle=F.l1_loss(x2_0_predict_cycle,real_data2)            
            errG_cycle = errG1_cycle + errG2_cycle            

            torch.autograd.set_detect_anomaly(True)
            
            errG = args.lambda_l1_loss*errG_cycle +  errG_adv + errG_cycle_adv + args.lambda_l1_loss*errG_L1
            errG.backward()
            
            optimizer_gen_diffusive_1.step()
            optimizer_gen_diffusive_2.step()
            optimizer_gen_non_diffusive_1to2.step()
            optimizer_gen_non_diffusive_2to1.step()           
            
            global_step += 1
            if iteration % 100 == 0:
                print('epoch {} iteration{}, G-Cycle: {}, G-L1: {}, G-Adv: {}, G-cycle-Adv: {}, G-Sum: {}, D Loss: {}, D_cycle Loss: {}'.format(epoch,iteration, errG_cycle.item(), errG_L1.item(),  errG_adv.item(), errG_cycle_adv.item(), errG.item(), errD.item(), errD_cycle.item()))
        
        if not args.no_lr_decay:
            
            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()
            scheduler_gen_non_diffusive_1to2.step()
            scheduler_gen_non_diffusive_2to1.step()
            scheduler_disc_diffusive_1.step()
            scheduler_disc_diffusive_2.step()

            scheduler_disc_non_diffusive_cycle1.step()
            scheduler_disc_non_diffusive_cycle2.step()

        if epoch % 10 == 0:
            torchvision.utils.save_image(x1_pos_sample, os.path.join(exp_path, 'xpos1_epoch_{}.png'.format(epoch)), normalize=True)
            torchvision.utils.save_image(x2_pos_sample, os.path.join(exp_path, 'xpos2_epoch_{}.png'.format(epoch)), normalize=True)
        #concatenate noise and source contrast
        x1_t = torch.cat((torch.randn_like(real_data1),real_data2),axis=1)
        fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)
        fake_sample1 = torch.cat((real_data2, fake_sample1),axis=-1)
        torchvision.utils.save_image(fake_sample1, os.path.join(exp_path, 'sample1_discrete_epoch_{}.png'.format(epoch)), normalize=True)
        pred1 = gen_non_diffusive_2to1(real_data2)
        #
        x2_t = torch.cat((torch.randn_like(real_data2), pred1),axis=1)
        fake_sample2_tilda = gen_diffusive_2(x2_t , t2, latent_z2)
        #
        pred1 = torch.cat((real_data2, pred1, gen_non_diffusive_1to2(pred1), fake_sample2_tilda[:,[0],:]),axis=-1)
        torchvision.utils.save_image(pred1, os.path.join(exp_path, 'sample1_translated_epoch_{}.png'.format(epoch)), normalize=True)


        x2_t = torch.cat((torch.randn_like(real_data2),real_data1),axis=1)
        fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)
        fake_sample2 = torch.cat((real_data1, fake_sample2),axis=-1)
        torchvision.utils.save_image(fake_sample2, os.path.join(exp_path, 'sample2_discrete_epoch_{}.png'.format(epoch)), normalize=True)
        pred2 = gen_non_diffusive_1to2(real_data1)
        #
        x1_t = torch.cat((torch.randn_like(real_data1), pred2),axis=1)
        fake_sample1_tilda = gen_diffusive_1(x1_t , t1, latent_z1)
        #
        pred2 = torch.cat((real_data1, pred2, gen_non_diffusive_2to1(pred2), fake_sample1_tilda[:,[0],:]),axis=-1)
        torchvision.utils.save_image(pred2, os.path.join(exp_path, 'sample2_translated_epoch_{}.png'.format(epoch)), normalize=True)

        """
        if args.save_content:
            if epoch % args.save_content_every == 0:
                print('Saving content.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                           'gen_diffusive_1_dict': gen_diffusive_1.state_dict(), 'optimizer_gen_diffusive_1': optimizer_gen_diffusive_1.state_dict(),
                           'gen_diffusive_2_dict': gen_diffusive_2.state_dict(), 'optimizer_gen_diffusive_2': optimizer_gen_diffusive_2.state_dict(),
                           'scheduler_gen_diffusive_1': scheduler_gen_diffusive_1.state_dict(), 'disc_diffusive_1_dict': disc_diffusive_1.state_dict(),
                           'scheduler_gen_diffusive_2': scheduler_gen_diffusive_2.state_dict(), 'disc_diffusive_2_dict': disc_diffusive_2.state_dict(),
                           'gen_non_diffusive_1to2_dict': gen_non_diffusive_1to2.state_dict(), 'optimizer_gen_non_diffusive_1to2': optimizer_gen_non_diffusive_1to2.state_dict(),
                           'gen_non_diffusive_2to1_dict': gen_non_diffusive_2to1.state_dict(), 'optimizer_gen_non_diffusive_2to1': optimizer_gen_non_diffusive_2to1.state_dict(),
                           'scheduler_gen_non_diffusive_1to2': scheduler_gen_non_diffusive_1to2.state_dict(), 'scheduler_gen_non_diffusive_2to1': scheduler_gen_non_diffusive_2to1.state_dict(),
                           'optimizer_disc_diffusive_1': optimizer_disc_diffusive_1.state_dict(), 'scheduler_disc_diffusive_1': scheduler_disc_diffusive_1.state_dict(),
                           'optimizer_disc_diffusive_2': optimizer_disc_diffusive_2.state_dict(), 'scheduler_disc_diffusive_2': scheduler_disc_diffusive_2.state_dict(),
                           'optimizer_disc_non_diffusive_cycle1': optimizer_disc_non_diffusive_cycle1.state_dict(), 'scheduler_disc_non_diffusive_cycle1': scheduler_disc_non_diffusive_cycle1.state_dict(),
                           'optimizer_disc_non_diffusive_cycle2': optimizer_disc_non_diffusive_cycle2.state_dict(), 'scheduler_disc_non_diffusive_cycle2': scheduler_disc_non_diffusive_cycle2.state_dict(),
                           'disc_non_diffusive_cycle1_dict': disc_non_diffusive_cycle1.state_dict(),'disc_non_diffusive_cycle2_dict': disc_non_diffusive_cycle2.state_dict()}

                torch.save(content, os.path.join(exp_path, 'content.pth'))

        if epoch % args.save_ckpt_every == 0:
            if args.use_ema:
                optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)
            torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1_{}.pth'.format(epoch)))
            torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2_{}.pth'.format(epoch)))
            torch.save(gen_non_diffusive_1to2.state_dict(), os.path.join(exp_path, 'gen_non_diffusive_1to2_{}.pth'.format(epoch)))
            torch.save(gen_non_diffusive_2to1.state_dict(), os.path.join(exp_path, 'gen_non_diffusive_2to1_{}.pth'.format(epoch)))
            if args.use_ema:
                optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)
        """

        if args.use_ema:
            optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
            optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
        gen_diffusive_1.eval()

        val_ts_psnr = numpy.zeros((len(val_data_t),), numpy.float32)
        val_ts_list = []
        with torch.no_grad():
            for i in range(len(val_data_t)):
                val_ts = numpy.zeros(val_data_t[i].shape, numpy.float32)
                used = numpy.zeros(val_data_t[i].shape, numpy.float32)
                for j in range(val_data_t[i].shape[0]):
                    val_patch_t = torch.tensor(val_data_t[i][j:j + 1, :, :], device=device).unsqueeze(0)

                    x1_t = torch.cat((torch.randn_like(val_patch_t), val_patch_t), axis=1)
                    # diffusion steps
                    fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)

                    val_ts[j, :, :] += fake_sample1.cpu().detach().numpy()[0, 0]
                    used[j, :, :] += 1

                assert used.min() > 0
                val_ts /= used

                ts_psnr = common_metrics.psnr(val_ts, val_data_s[i])

                val_ts_psnr[i] = ts_psnr
                val_ts_list.append(val_ts)

        print(val_ts_psnr)
        msg = "Epoch:%d  val_ts_psnr:%f/%f" % (epoch, val_ts_psnr.mean(), val_ts_psnr.std())
        gen_images_test = numpy.concatenate([val_data_s[0], val_ts_list[0], val_data_t[0]], 2)
        gen_images_test = numpy.expand_dims(gen_images_test, 0).astype(numpy.float32)
        gen_images_test = common_cmf.generate_display_image(gen_images_test, is_seg=False)
        skimage.io.imsave(os.path.join(exp_path, "gen_images_test.jpg"), gen_images_test)

        if val_ts_psnr.mean() > best_psnr:
            best_psnr = val_ts_psnr.mean()

            torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1_best.pth'))
            torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2_best.pth'))

        msg += "  best_ts_psnr:%f" % best_psnr
        print(msg)

        torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1_last.pth'))
        torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2_last.pth'))
        if args.use_ema:
            optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
            optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
        
        gen_diffusive_1.train()


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    
    parser.add_argument('--resume', action='store_true',default=False)
    
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_channels', type=int, default=1,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='ixi_synth', help='name of experiment')
    parser.add_argument('--input_path', default="/home/chenxu/datasets/cmf", help='path to input data')
    parser.add_argument('--output_path', help='path to output saves')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    
    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save ckpt every x epochs')
    parser.add_argument('--lambda_l1_loss', type=float, default=0.5, help='weightening of l1 loss part of diffusion ans cycle models')
   
    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--contrast1', type=str, default='T1',
                        help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='T2',
                        help='contrast selection for model')
    parser.add_argument('--port_num', type=str, default='6021',
                        help='port selection for code')
    parser.add_argument('--gpu', type=int, default=0, help='debug')
    parser.add_argument('--debug', type=int, default=0, help='debug')
   
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    train_syndiff(device, args)
