
import argparse
import torch
import numpy as np, h5py
import numpy
import os
import torch.optim as optim
import torchvision
from backbones.ncsnpp_generator_adagn import NCSNpp
from dataset import CreateDatasetSynthesis

import torch.nn.functional as F

import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as SSIM
import pdb

if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_pelvic_pt as common_pelvic

def psnr(img1, img2):
    #Peak Signal to Noise Ratio

    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(img1.max() / torch.sqrt(mse))

        
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


def load_checkpoint(checkpoint_dir, netG, name_of_network, epoch,device = 'cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)  

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint
   
    for key in list(ckpt.keys()):
         ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()
#%%
def sample_and_test(args):
    torch.manual_seed(42)
    # device = 'cuda:0'
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu))
    epoch_chosen=args.which_epoch

    #loading dataset
    phase='test'

    test_data_s, test_data_t, _, _ = common_pelvic.load_val_data(args.input_path, valid=True)

    #Initializing and loading network
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)

    checkpoint_file = os.path.join(checkpoint_path, "{}_{}.pth")
    load_checkpoint(checkpoint_file, gen_diffusive_1,'gen_diffusive_1',epoch=str(epoch_chosen), device = device)
    load_checkpoint(checkpoint_file, gen_diffusive_2,'gen_diffusive_2',epoch=str(epoch_chosen), device = device)

    T = get_time_schedule(args, device)
    
    pos_coeff = Posterior_Coefficients(args, device)

    if args.output_path and not os.path.exists(save_dir):
        os.makedirs(args.output_path)

    test_st_psnr = numpy.zeros((len(test_data_t),), numpy.float32)
    test_ts_psnr = numpy.zeros((len(test_data_t),), numpy.float32)
    test_st_ssim = numpy.zeros((len(test_data_t),), numpy.float32)
    test_ts_ssim = numpy.zeros((len(test_data_t),), numpy.float32)
    test_st_mae = numpy.zeros((len(test_data_t),), numpy.float32)
    test_ts_mae = numpy.zeros((len(test_data_t),), numpy.float32)
    with torch.no_grad():
        for i in range(len(test_data_t)):
            syn_im = numpy.zeros(test_data_t[i].shape, numpy.float32)
            used = numpy.zeros(test_data_t[i].shape, numpy.float32)
            for j in range(test_data_t[i].shape[0]):
                input_patch = torch.tensor(test_data_t[i][j:j + 1, :, :], device=device).unsqueeze(0)

                x1_t = torch.cat((torch.randn_like(input_patch), input_patch), axis=1)
                # diffusion steps
                fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)

                syn_im[j, :, :] += fake_sample1.cpu().detach().numpy()[0, 0]
                used[j, :, :] += 1

            assert used.min() > 0
            syn_im /= used

            test_ts_psnr[i] = common_metrics.psnr(syn_im, test_data_s[i])
            test_ts_ssim[i] = SSIM(syn_im, test_data_s[i], data_range=2.)
            test_ts_mae[i] = abs(common_pelvic.restore_hu(syn_im) - common_pelvic.restore_hu(test_data_s[i])).mean()
            if args.output_path:
                common_pelvic.save_nii(syn_im, "syn_ts_%d.nii.gz" % i)

        for i in range(len(test_data_s)):
            syn_im = numpy.zeros(test_data_s[i].shape, numpy.float32)
            used = numpy.zeros(test_data_s[i].shape, numpy.float32)
            for j in range(test_data_s[i].shape[0]):
                input_patch = torch.tensor(test_data_s[i][j:j + 1, :, :], device=device).unsqueeze(0)

                x1_t = torch.cat((torch.randn_like(input_patch), input_patch), axis=1)
                # diffusion steps
                fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x1_t, T, args)

                syn_im[j, :, :] += fake_sample1.cpu().detach().numpy()[0, 0]
                used[j, :, :] += 1

            assert used.min() > 0
            syn_im /= used

            test_st_psnr[i] = common_metrics.psnr(syn_im, test_data_t[i])
            test_st_ssim[i] = SSIM(syn_im, test_data_t[i], data_range=2.)
            test_st_mae[i] = abs(common_pelvic.restore_hu(syn_im) - common_pelvic.restore_hu(test_data_t[i])).mean()
            if args.output_path:
                common_pelvic.save_nii(syn_im, "syn_st_%d.nii.gz" % i)

    msg = ("test_st_psnr:%f/%f  test_st_ssim:%f/%f  test_st_mae:%f/%f  test_ts_psnr:%f/%f  test_ts_ssim:%f/%f  test_ts_mae:%f/%f\n" %
           (test_st_psnr.mean(), test_st_psnr.std(), test_st_ssim.mean(), test_st_ssim.std(), test_st_mae.mean(), test_st_mae.std(),
            test_ts_psnr.mean(), test_ts_psnr.std(), test_ts_ssim.mean(), test_ts_ssim.std(), test_ts_mae.mean(), test_ts_mae.std()))
    print(msg)
    if args.output_path:
        with open(os.path.join(args.output_path, "result.txt"), "w") as f:
            f.write(msg)

        numpy.save(os.path.join(args.output_path, "st_psnr.npy"), test_st_psnr)
        numpy.save(os.path.join(args.output_path, "ts_psnr.npy"), test_ts_psnr)
        numpy.save(os.path.join(args.output_path, "st_ssim.npy"), test_st_ssim)
        numpy.save(os.path.join(args.output_path, "ts_ssim.npy"), test_ts_ssim)
        numpy.save(os.path.join(args.output_path, "st_mae.npy"), test_st_mae)
        numpy.save(os.path.join(args.output_path, "ts_mae.npy"), test_ts_mae)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
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
    parser.add_argument('--input_path', default="/home/chenxu/datasets/pelvic/h5_data_nonrigid", help='path to input data')
    parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint files')
    parser.add_argument('--output_path', type=str, default="", help='path to outputs')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='sample generating batch size')
    
    #optimizaer parameters    
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--contrast1', type=str, default='T1',
                        help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='T2',
                        help='contrast selection for model')
    parser.add_argument('--which_epoch', type=str, default="best", choices=["best", "last"])
    parser.add_argument('--gpu', type=int, default=0)


    parser.add_argument('--source', type=str, default='T2',
                        help='source contrast')   
    args = parser.parse_args()
    
    sample_and_test(args)
    
