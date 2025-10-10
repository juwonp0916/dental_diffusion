import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

from collections import OrderedDict
import re

import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_generation import PVCNN2Base
import torch.distributed as dist
from datasets.dataset_generation import ToothDataset, FDIS
from datasets.augmentation_techniques import *
from torchvision import transforms
import json
from tqdm import tqdm

'''
----- Some utilities -----
'''

def norm(v, f):
    v = (v - v.min()) / (v.max() - v.min()) - 0.5
    return v, f

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm

def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

''' 
----- Models ----- 
'''

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2) ** 2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus) * 1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min, torch.ones_like(cdf_min) * 1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < 0.001, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                    torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta) * 1e-12))))
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        self.posterior_mean_coef3 = (
            1.0 + ((torch.sqrt(self.alphas_cumprod) - 1.) * ( torch.sqrt(self.alphas_cumprod_prev) + torch.sqrt(alphas)))
            / (1.0 - self.alphas_cumprod))

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])

        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """ Diffuse the data (t == 0 means diffused for 1 step) """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)

        assert noise.shape == x_start.shape

        return (self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """ Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0) """
        assert x_start.shape == x_t.shape
        posterior_mean = (self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                          self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t,
                                                       x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])

        return posterior_mean, posterior_variance, posterior_log_variance_clipped
                                                                        
    def p_mean_variance(self, denoise_fn, xt, model_kwargs, t, return_pred_xstart: bool):

        model_output, saved_attns  = denoise_fn(xt, 
                                    t, 
                                    model_kwargs)

        device = xt.device
        shape = xt.shape

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:  
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(device)),
                'fixedsmall': (self.posterior_variance.to(device),
                               self.posterior_log_variance_clipped.to(device))}[self.model_var_type]

            model_variance = self._extract(model_variance, t, shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(model_log_variance, t, shape) * torch.ones_like(model_output)

        else:
            raise NotImplementedError(self.model_var_type)



        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(xt, t=t, eps=model_output)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=xt, t=t)

        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape
        assert model_variance.shape == model_log_variance.shape

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon, saved_attns
        else:
            return model_mean, model_variance, model_log_variance, saved_attns

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps)
    
    ''' 
    ----- Sampling ----- 
    '''

    def p_sample(self, denoise_fn, xt, model_kwargs, t, noise_fn, return_pred_xstart=False):
        """ Sample from the model """

        model_mean, _, model_log_variance, pred_xstart, saved_attns = self.p_mean_variance(denoise_fn, xt=xt, model_kwargs=model_kwargs, t=t,
                                                                              return_pred_xstart=True)
    

        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)

        # no noise when t == 0 
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [xt.shape[0]] + [1] * (len(model_mean.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        return (sample, pred_xstart, saved_attns) if return_pred_xstart else (sample, saved_attns)

    def p_sample_loop(self, model_kwargs, denoise_fn,  noise_fn=torch.randn, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps
        """
        x0 = model_kwargs['x0']
        B = x0.shape[0]
        device = x0.device
        img = noise_fn(size=x0.shape, dtype=torch.float, device=device)

        for t in tqdm(reversed(range(0, self.num_timesteps if not keep_running else len(self.betas)))):

            t_ = torch.empty(B, dtype=torch.int64, device=device).fill_(t)

            img, _ = self.p_sample(denoise_fn=denoise_fn, xt=img, model_kwargs = model_kwargs, t=t_, noise_fn=noise_fn, return_pred_xstart=False)

        return img

    ''' 
    ----- Losses ----- 
    '''

    def mse_mean_flat(self, B, noise, eps_recon, mask):
        """
        Take the mean over all non-batch dimensions, considering only unmasked elements.
        """
        total_loss = 0

        for b in range(B):

            gt_noise = noise[b][mask[b].view(-1).bool()]
            pred_noise = eps_recon[b][mask[b].view(-1).bool()]
            loss_b = ((gt_noise - pred_noise)**2).mean()
            total_loss = total_loss + loss_b

        return total_loss / B

    def p_losses(self, denoise_fn, t, noise, model_kwargs):
        
        data_start = model_kwargs['x0']

        B = data_start.shape[0]
        assert t.shape == torch.Size([B])

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':

            eps_recon, _  = denoise_fn(data_t, 
                                       t, 
                                       model_kwargs)
    
            losses = self.mse_mean_flat(B, noise, eps_recon, model_kwargs['l_mask']) # only calculate loss on target, ignore context teetn

        elif self.loss_type == 'kl':
            pass
            # losses = self._vb_terms_bpd(
            #     denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
            #     return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        return losses



class PVCNN2(PVCNN2Base):

    num_n = 128 # Number of neighbors

    sa_blocks = [
        ((32, 2, 32), (512, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att, dropout, extra_feature_channels=3,
                 width_multiplier=1.0, voxel_resolution_multiplier=1.0):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
                         dropout=dropout, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str,
                 width_mult: float, vox_res_mult: float):
        super(Model, self).__init__()

        # Create diffusion
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        # Create point-voxel-cnn network
        self.model = PVCNN2(num_classes=args.nc, embed_dim=args.embed_dim,
                            use_att=args.attention, dropout=args.dropout, extra_feature_channels=args.extra_feature_nc,
                            width_multiplier=width_mult, voxel_resolution_multiplier=vox_res_mult)

    def _denoise(self, xt, t, model_kwargs):
        B = xt.shape[0]

        assert xt.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        # Extract fdi_indices as positional argument (required by new model signature)
        fdi_indices = model_kwargs['fdi_indices']
        out, attn_mask = self.model(xt, t, model_kwargs['x0'], fdi_indices,
                                     model_kwargs['l_mask'], model_kwargs['o_mask'],
                                     model_kwargs.get('bound'))

        return out, attn_mask

    def get_loss_iter_teethmask(self, noise_batch, model_kwargs):

        dentition_points = model_kwargs['x0']
        B = dentition_points.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=noise_batch.device)
        
        losses = self.diffusion.p_losses(denoise_fn=self._denoise, 
                                         t=t, 
                                         noise=noise_batch,
                                         model_kwargs=model_kwargs)

        return losses

    def gen_samples(self, model_kwargs, noise_fn = torch.randn, keep_running=False):

        out = self.diffusion.p_sample_loop(model_kwargs, self._denoise, noise_fn=noise_fn,
                                             keep_running=keep_running)

        return out

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)

def validate_training_scenario(latent_mask, obs_mask, original_missing_mask, batch_idx, logger):
    """
    Validate that the training scenario is realistic:
    1. We're not trying to reconstruct naturally missing teeth
    2. We have sufficient context teeth for meaningful training
    3. The latent and observation masks are consistent
    """
    B = latent_mask.shape[0]

    for b in range(B):
        # Check 1: Ensure latent teeth were originally present
        if original_missing_mask is not None:
            latent_indices = torch.where(latent_mask[b, :, 0, 0] == 1)[0]
            originally_missing_indices = torch.where(original_missing_mask[b])[0]

            invalid_latent = torch.isin(latent_indices, originally_missing_indices)
            if torch.any(invalid_latent):
                invalid_count = torch.sum(invalid_latent).item()
                if logger:
                    logger.warning(f"Batch {batch_idx}, sample {b}: Trying to reconstruct {invalid_count} naturally missing teeth")
                return False

        # Check 2: Ensure we have sufficient context teeth
        obs_count = torch.sum(obs_mask[b, :, 0, 0]).item()
        latent_count = torch.sum(latent_mask[b, :, 0, 0]).item()

        if obs_count < 2:  # Need at least 2 context teeth
            if logger:
                logger.warning(f"Batch {batch_idx}, sample {b}: Insufficient context teeth ({obs_count})")
            return False

        if latent_count == 0:  # Need at least 1 tooth to reconstruct
            if logger:
                logger.warning(f"Batch {batch_idx}, sample {b}: No teeth to reconstruct")
            return False

        # Check 3: Ensure masks are consistent (no overlap)
        overlap = torch.sum(latent_mask[b, :, 0, 0] * obs_mask[b, :, 0, 0]).item()
        if overlap > 0:
            if logger:
                logger.warning(f"Batch {batch_idx}, sample {b}: Latent and observation masks overlap")
            return False

    return True

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)

    elif schedule_type == 'warm0.1':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.2':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.5':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    else:
        raise NotImplementedError(schedule_type)

    return betas



def get_dataset(path, tooth_npoints, mode='train'):

    dataset = ToothDataset(
                                path = path,
                                mode=mode, 
                                tooth_npoints = tooth_npoints,
                                aug_transforms=transforms.Compose(
                                    [   
                                        RandomMirror(p=0.5),
                                        RandomRotate(angle=[-1 / 48, 1 / 48], axis="x", p=0.5),
                                        RandomRotate(angle=[-1 / 48, 1 / 48], axis="y", p=0.5),
                                        RandomRotate(angle=[-1 / 48, 1 / 48], axis="z", p=0.5),
                                        RandomScale(scale = [0.95,1.05], p=0.75),
                                        ShufflePoint(p=0.9)
                                    ] 
                )) if mode == 'train' else ToothDataset(
                                path = path,
                                mode=mode, 
                                tooth_npoints = tooth_npoints,
                                aug_transforms=None
                        )

    print(f'How many training dataset? : {len(dataset)}')

    return dataset


def get_dataloader(opt, dataset, local_rank, mode = 'train'):

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=opt.world_size,
        rank=local_rank,
        shuffle = (mode=='train')
    )

    dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=opt.bs, 
                                                sampler=sampler,
                                                shuffle=sampler is None, 
                                                num_workers=int(opt.workers),
                                                pin_memory = False,
                                                persistent_workers = False,  # Reduced memory usage
                                                drop_last=False)
    
    return dataloader, sampler


def generate_val_samples(opt, model, val_dataset, outf_syn, epoch, device):
    logger.info('Generation: eval')

    model.eval()

    sample_batch_size = 1 # how many validation samples to generate
    val_sample_list = val_dataset.sample_patient(sample_batch_size)
    val_dentition_points = torch.stack([sample['dentition_points'] for sample in val_sample_list], 0).to(device)
    val_fdi_indices = torch.stack([sample['fdi_indices'] for sample in val_sample_list], 0).to(device)  # NEW
    val_bound = None
    if 'bounds_cyl' in val_sample_list[0]:
        val_bound = torch.stack([sample['bounds_cyl'] for sample in val_sample_list], 0).to(device)
    val_dentition_ids = [sample['patient_id'] for sample in val_sample_list]

    # Handle original missing teeth in validation
    val_original_missing_mask = None
    if 'original_missing_mask' in val_sample_list[0]:
        val_original_missing_mask = torch.stack([sample['original_missing_mask'] for sample in val_sample_list], 0).to(device)

    latent_mask_val = torch.zeros_like(val_dentition_points[:,:,:1,:1]).to(device)

    for i in range(sample_batch_size):
        # Get existing teeth for validation sampling
        if val_original_missing_mask is not None:
            existing_teeth_indices = torch.where(~val_original_missing_mask[i])[0]
        else:
            existing_teeth_indices = torch.arange(val_dentition_points.shape[1], device=device)

        if len(existing_teeth_indices) == 0:
            continue  # Skip if no teeth exist

        # Select fewer teeth for validation to get cleaner results
        n_missing_val = min(random.randint(1, min(3, opt.max_missing_teeth)), len(existing_teeth_indices))

        # Prefer non-wisdom teeth for validation as well
        non_wisdom_existing = []
        for idx in existing_teeth_indices:
            fdi = FDIS[idx] if idx < len(FDIS) else 0
            if fdi % 10 != 8:  # Not a wisdom tooth
                non_wisdom_existing.append(idx)

        if len(non_wisdom_existing) >= n_missing_val:
            selected_existing = torch.tensor(non_wisdom_existing, device=device)[torch.randperm(len(non_wisdom_existing))[:n_missing_val]]
        else:
            selected_existing = existing_teeth_indices[torch.randperm(len(existing_teeth_indices))[:n_missing_val]]

        latent_mask_val[i, selected_existing, 0, 0] = 1

    obs_mask_val = torch.ones_like(latent_mask_val) - latent_mask_val

    # Ensure we don't try to observe naturally missing teeth in validation
    if val_original_missing_mask is not None:
        obs_mask_val = obs_mask_val * (~val_original_missing_mask.unsqueeze(-1).unsqueeze(-1))

    val_data_dict = {
        'x0': val_dentition_points,
        'fdi_indices': val_fdi_indices,  # NEW: Pass actual FDI numbers
        'l_mask': latent_mask_val,
        'o_mask':obs_mask_val,
        'bound':val_bound
    }

    # generate some samples
    with torch.no_grad():

        sample_out = model.gen_samples(model_kwargs=val_data_dict).detach().cpu()
        val_dentition_points = val_dentition_points.detach().cpu()
        obs_mask_val = obs_mask_val.detach().cpu()
        latent_mask_val = latent_mask_val.detach().cpu()

        for i in range(sample_batch_size):

            missing_idx = np.where(latent_mask_val.numpy().squeeze())[0]

            condition_points = val_dentition_points[i][obs_mask_val[i].view(-1).bool()]
            generated_points = sample_out[i][latent_mask_val[i].view(-1).bool()]
            ground_truth_points = val_dentition_points[i][latent_mask_val[i].view(-1).bool()]
            patient_id = val_dentition_ids[i]

            export_dentition_npy('{}_{}_condition'.format(patient_id,missing_idx), '%s/epoch_%03d' % (outf_syn, epoch), condition_points.numpy())
            export_dentition_npy('{}_{}_gen'.format(patient_id,missing_idx), '%s/epoch_%03d' % (outf_syn, epoch), generated_points.numpy())
            export_dentition_npy('{}_{}_gt'.format(patient_id,missing_idx), '%s/epoch_%03d' % (outf_syn, epoch), ground_truth_points.numpy())

    model.train()


def remove_module_prefix(state_dict):

    new_state_dict = OrderedDict()
    pattern = re.compile('module.')

    for k,v in state_dict.items():
        if re.search("module", k):
            new_state_dict[re.sub(pattern, '', k)] = v    
        else:
            new_state_dict[k] = v

    return new_state_dict


def train(local_rank, opt, output_dir):
    logger = setup_logging(output_dir)

    # DDP 관련 
    take_action = local_rank == 0
    # we only need one process to do auxillary stuff during training. 

    if take_action:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')
        logger.info(opt) # print options


    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=opt.world_size, rank=local_rank)

    opt.bs = int(opt.bs / opt.ngpus_per_node)
    opt.workers = int(opt.workers / opt.ngpus_per_node)


    ''' Dataset and data loader '''
    train_dataset = get_dataset(path = opt.path, tooth_npoints = opt.tooth_npoints, mode='train')
    val_dataset = get_dataset(path = opt.path, tooth_npoints = opt.tooth_npoints, mode='val')

    dataloader, sampler = get_dataloader(opt, train_dataset,local_rank, mode = 'train')


    ''' Create networks '''
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type, opt.width_mult, opt.vox_res_mult)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)  


    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # Memory optimization settings
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.9)

    model = model.to(device)


    if opt.model != '': # if continue train
        ckpt = torch.load(opt.model, map_location=device)
        try:
            model.load_state_dict(ckpt['model_state'])
            logger.info(f'RANK {local_rank}: saved weight loaded!')

        except:
            model.load_state_dict(remove_module_prefix(ckpt['model_state']))
            logger.info(f'RANK {local_rank}: saved weight loaded after remove module prefix!')

        optimizer.load_state_dict(ckpt['optimizer_state'])
        logger.info(f'RANK {local_rank}: optimizer status loaded')
        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 0


    def _transform_(m):
        return nn.parallel.DistributedDataParallel(
            m, device_ids=[local_rank], output_device=local_rank)

    model.multi_gpu_wrapper(_transform_)



    lr_decay_epochs = [(opt.niter//4), (opt.niter//4)*2, (opt.niter//4)*3] # Decay learning rate by 0.25, 3 times throughout training


    for epoch in range(start_epoch, opt.niter):

        sampler.set_epoch(epoch)

        logger.info(f'RANK {local_rank}: Training start at epoch {epoch}')

        
        for i, data in enumerate(dataloader):

            dentition_points = data['dentition_points'].to(device) # (b, K, 3, 1024)
            fdi_indices = data['fdi_indices'].to(device)  # NEW: (b, 28) actual FDI numbers
            bound = data.get('bounds_cyl')  # Optional, may not be in dataset
            if bound is not None:
                bound = bound.to(device)  # (b, 28, 5)
            original_missing_mask = data['original_missing_mask'].to(device) if 'original_missing_mask' in data else None
            B, nT, nD, nP = dentition_points.shape

            latent_mask = torch.zeros_like(dentition_points[:,:,:1,:1]).to(device)

            for b in range(B):
                # Get existing teeth for this batch item
                if original_missing_mask is not None:
                    existing_teeth_indices = torch.where(~original_missing_mask[b])[0]
                else:
                    # Fallback: assume all teeth exist (legacy behavior)
                    existing_teeth_indices = torch.arange(nT, device=device)

                if len(existing_teeth_indices) == 0:
                    continue  # Skip if no teeth exist

                # Randomly select number of teeth to simulate as missing from existing teeth only
                n_missing = min(random.randint(1, opt.max_missing_teeth), len(existing_teeth_indices))

                # Select random subset of existing teeth to simulate as missing
                selected_existing = existing_teeth_indices[torch.randperm(len(existing_teeth_indices))[:n_missing]]

                # Avoid selecting wisdom teeth (FDI ending in 8) for simulation if possible
                non_wisdom_existing = []
                wisdom_existing = []
                for idx in existing_teeth_indices:
                    fdi = FDIS[idx] if idx < len(FDIS) else 0
                    if fdi % 10 == 8:  # Wisdom tooth (ends in 8)
                        wisdom_existing.append(idx)
                    else:
                        non_wisdom_existing.append(idx)

                # Prefer selecting non-wisdom teeth for simulation
                if len(non_wisdom_existing) >= n_missing:
                    selected_existing = torch.tensor(non_wisdom_existing, device=device)[torch.randperm(len(non_wisdom_existing))[:n_missing]]

                latent_mask[b, selected_existing, 0, 0] = 1

            obs_mask = torch.ones_like(latent_mask) - latent_mask

            # Ensure we don't try to observe naturally missing teeth
            if original_missing_mask is not None:
                # Set obs_mask to 0 for naturally missing teeth
                obs_mask = obs_mask * (~original_missing_mask.unsqueeze(-1).unsqueeze(-1))

            # Validate training scenario before proceeding
            if not validate_training_scenario(latent_mask, obs_mask, original_missing_mask, i, logger if take_action else None):
                continue  # Skip this batch if validation fails

            # IMPORTANT
            # obs_mask (observed mask): mask that indicates which teeth are existing context teeth
            # latent_mask: mask that indicates which teeth are simulated for omission and trying to reconstruct back

            noise_batch = torch.randn_like(dentition_points).to(device)

            data_dict = {
                'x0': dentition_points,
                'fdi_indices': fdi_indices,  # NEW: Pass actual FDI numbers
                'l_mask': latent_mask,
                'o_mask':obs_mask,
                'bound':bound
            }
            loss = model.get_loss_iter_teethmask(noise_batch, model_kwargs=data_dict)

            # Optimize network parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clear memory after each batch
            if i % 5 == 0:  # Clear cache every 5 batches
                torch.cuda.empty_cache()

            # Print progress
            if i % opt.print_freq == 0 and take_action:
                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                            .format(epoch, opt.niter, i, len(dataloader), loss.item()))
                
        if (epoch+1) in lr_decay_epochs:
            for param_group in optimizer.param_groups:
                logger.info('old lr: {}'.format(param_group['lr']))
                param_group['lr'] = param_group['lr'] * opt.lr_decay_factor
                logger.info('new lr: {}'.format(param_group['lr']))

        # Visualize some samples during training
        if ((epoch + 1) % opt.vizIter == 0 and take_action):
            generate_val_samples(opt, model, val_dataset, outf_syn, epoch, device)

        # Intermediate saving checkpoint
        if (epoch + 1) % opt.saveIter == 0:
            if take_action:
                save_dict = {'epoch': epoch,
                             'model_state': model.state_dict(),
                             'optimizer_state': optimizer.state_dict()
                             }

                torch.save(save_dict, '%s/epoch_%d.pth' % (output_dir, epoch))

            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            model.load_state_dict(
                torch.load('%s/epoch_%d.pth' % (output_dir, epoch), map_location=map_location)['model_state'])


    dist.destroy_process_group()


def main():
    opt = parse_args()
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9991'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)

    output_dir = get_output_dir(dir_id, exp_id)

    with open(os.path.join(output_dir,'options.txt'),'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    copy_source(__file__, output_dir)

    # initiate distributed training
    opt.ngpus_per_node = torch.cuda.device_count()
    opt.world_size = opt.ngpus_per_node

    print(f'DDP World_size:{opt.world_size}')

    mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data', help="set the path to the dataset here")

    # Data loader parameters
    parser.add_argument('--bs', type=int, required=True, help='input batch size, 28teeth/1024points 기준 1batch=1gpu')
    parser.add_argument('--workers', type=int, required=True,  help='workers dataloader, 1batch=1worker as starting point')
    parser.add_argument('--niter', type=int, required=True, help='number of epochs to train for')

    # Input point cloud
    parser.add_argument('--nc', type=int, default=3, help="dimension of one point (usually 3 for x, y,z)")
    parser.add_argument('--extra_feature_nc', type=int, default=8, help="1 for binary mask + 7 for deterministic FDI encoding (jaw/quadrant/position)")

    parser.add_argument('--max_missing_teeth', type=int, default=6, help="maximum missing teeth to simulate") 
    parser.add_argument('--tooth_npoints', type=int, default=1024, help="num points per toorh") 

    ''' Model '''
    # Diffusion process parameters (variance schedule, number of steps)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--schedule_type', type=str, default='linear')
    parser.add_argument('--time_num', type=int, default=1000, help='number of timesteps T in diffusion process')
    # parser.add_argument('--debug', action='store_true')

    # Model parameters
    parser.add_argument('--attention', type=eval, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--model_mean_type', type=str, default='eps')
    parser.add_argument('--model_var_type', type=str, default='fixedsmall')
    parser.add_argument('--vox_res_mult', type=float, default=1.0)
    parser.add_argument('--width_mult', type=float, default=1.0)

    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=1, help='lr decay for EBM')
    parser.add_argument('--lr_decay_factor', type=float, default=0.45, help='')


    # Model path (for continuing the training of existing models)
    parser.add_argument('--model', default='', help="path to model (to continue training)")

    parser.add_argument('--saveIter', type=int, default=50, help='unit: epoch')
    parser.add_argument('--vizIter', type=int, default=100, help='unit: epoch')
    parser.add_argument('--print_freq', type=int, default=10, help='unit: iter')

    # Manual seed for deterministic sampling, etc.
    # TODO apply manual seed for reproducibility using "set_seed" in file_utils.py
    parser.add_argument('--manualSeed', default=1234, type=int, help='random seed')

    # Parse arguments
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
