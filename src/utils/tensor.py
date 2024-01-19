from numpy import shape
import torch
import copy

def mgrid(*tensors1D):
    reshape_dim =tuple([len(tensors1D)] + [tensors1D[i].shape[0] for i in range(len(tensors1D))])
    permute_dims = [i for i in range(len(tensors1D)+1)]
    permute_dims = tuple(permute_dims[1:]+[permute_dims[0]])

    cart_prod_xyz=torch.cartesian_prod(*tensors1D).T
    
    return torch.permute(cart_prod_xyz.reshape(reshape_dim),dims=permute_dims)
    
def mgrid_l2(tmgrid):
    return torch.sqrt( torch.einsum('...i,...i->...',tmgrid,torch.conj(tmgrid)))

def linspace(start,end,steps):
    assert start.shape==end.shape

    alpha = torch.linspace(0,1,steps,dtype=torch.float64)
    return start+torch.einsum('i,...->i...',alpha,end-start)
    
def arange(start, step, steps):
    return linspace(start=start,end = steps*step+start,steps=steps+1)

