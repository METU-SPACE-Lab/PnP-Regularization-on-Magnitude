import torch
from torchmetrics import StructuralSimilarityIndexMeasure as _SSIM
import numpy as np
import random

# CUSTOM MISC
def maxmag3D(x):
    m=torch.abs(x)
    return torch.max_pool3d(m,kernel_size=x.shape[-3:])

# MISC
def _exclude_dims(nof_dims, dims_exclude):
    if type(dims_exclude)==int:
        dims_exclude=[dims_exclude]
    dims_use=list(range(nof_dims))
    
    for dim_exclude in dims_exclude:
        dim_exclude=dim_exclude % nof_dims
        dims_use.remove(dim_exclude)
    
    return dims_use

# STATISTICS
def std(x,dims_exclude=[0]):
    nof_dims = len(x.shape)
    dims_use = _exclude_dims(nof_dims=nof_dims, dims_exclude=dims_exclude)
    return torch.std(x,dim=dims_use, keepdim=True)


def l22(x,dims_exclude=[0]):
    nof_dims = len(x.shape)
    dims_use = _exclude_dims(nof_dims=nof_dims, dims_exclude=dims_exclude)

    return torch.sum(torch.abs(x)**2,dim=dims_use, keepdim=True)

def l2(x,dims_exclude=[0]):
    return torch.sqrt(l22(x,dims_exclude))

def power(x,dims_exclude=[0]):
    nof_dims = len(x.shape)
    dims_use = _exclude_dims(nof_dims=nof_dims, dims_exclude=dims_exclude)
    return torch.mean(torch.abs(x)**2, dim=dims_use, keepdim=True)

def snr(signal,noise,dims_exclude=[0]):
    return power(signal,dims_exclude)/power(noise,dims_exclude)

def snrdB(signal,noise,dims_exclude=[0]):
    return lin2dB(snr(signal,noise,dims_exclude))

def lin2dB(x):
    if type(x)!=torch.Tensor:
        x=torch.tensor(x)
    return 10*torch.log10(x)

def dB2lin(x_dB):
    if type(x_dB)!=torch.Tensor:
        x_dB=torch.tensor(x_dB)
    return 10**(x_dB/10)

def snr2npower(snr, signal_power):
    return signal_power/snr
    
def wgn(size,s=1,dtype=torch.complex128):
    return s*torch.randn(size=size,dtype=dtype)

def awgn(signal,snrdB,dims_exclude=[0],dtype=torch.complex128):
    sn_power_ratio = dB2lin(snrdB)
    var = snr2npower(sn_power_ratio,power(signal,dims_exclude))
    return signal+wgn(size=signal.shape,s=torch.sqrt(var),dtype=dtype)

def snrdB2nvar(x_clean,snrdB):
    sn_power_ratio = dB2lin(snrdB)
    n_var = snr2npower(sn_power_ratio,power(x_clean,dims_exclude=[0]))
    return n_var

# METRICS
def mse(target,estimation,dims_exclude=[0]):
    return power(target-estimation,dims_exclude=dims_exclude)

def psnr3D(target, estimation):
    vox_max = maxmag3D(target)
    return lin2dB((vox_max**2)/mse(target,estimation,dims_exclude=0))

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
