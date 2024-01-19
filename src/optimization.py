import torch
import torch.nn as nn
import numpy as np

from src.utils.stats import l2 as l2_norm
from src.utils.stats import _exclude_dims,maxmag3D,psnr3D

import src.utils.tensor as tensor
from src.forward_models.bornapprox import C_SOL

import itertools
import warnings

from tqdm import tqdm as tqdm

def cg(Q, b, x, inner_prod, max_iter=100,th= 1e-10):
    r = b - Q(x)
    p = r
    rsold = inner_prod(r,r) #r' @ r;
    x_old=x
    
    for i in range(max_iter):
        Qp = Q(p)
        alpha = rsold / inner_prod(p, Qp) # rsold / p'@ Ap
        print(alpha.shape)
        x = x + alpha * p
        r = r - alpha * Qp
        rsnew = inner_prod(r, r)#r' @ r;

        if torch.all((l2_norm(x-x_old)/l2_norm(x_old)) < th ):
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew
        x_old = x

    return x

def cg3D(Q, b, x, max_iter=100,th= 1e-10,tqdm_active=False):
    inner_prod = lambda x1,x2: torch.einsum('...xyz,...xyz->...',torch.conj(x1),x2)
    scale = lambda constant,vec: torch.einsum('i...,i...->i...',constant, vec)
    
    r = b - Q(x)
    p = r
    rsold = inner_prod(r,r) #r' @ r;
    x_old=x
    
    if not torch.all(r==0):
        
        for i in tqdm(range(max_iter),disable=not tqdm_active):
            Qp = Q(p)
            alpha = rsold / inner_prod(p, Qp) # rsold / p'@ Ap
            
            x = x + scale(alpha,p) 
            r = r - scale(alpha, Qp)
            rsnew = inner_prod(r, r) #r' @ r;

            if torch.all((l2_norm(x-x_old)/l2_norm(x_old)) < th ):
                break

            p = r + scale((rsnew / rsold),  p)
            rsold = rsnew
            x_old = x

    return x

class MDGrid():
    def __init__(self) -> None:
        self._search_space={}
        self._keys=[]
        self._dim_per_key=[]
        self._iter_idxs=[]
        self.iter_count=0
        self.state_list=[]
    def entry(self,*args):
        return list(args)

    def set(self,**kwargs):
        self.reset()

        self._keys=list(kwargs.keys())

        for key in self._keys:
            value=kwargs[key]
            
            if type(value)!=list:
                value=[value]

            self._search_space[key]=value
            self._dim_per_key.append(len(value))
        
        iter_ranges=[range(i) for i in self._dim_per_key]
        self._iter_idxs = list(itertools.product(*iter_ranges))
        
    def reset(self):
        self._search_space={}
        self._keys=[]
        self._dim_per_key=[]
        self._iter_idxs=[]
        self.iter_count=0
        
    
    def map(self,gridpass_f):
        for idxs in self._iter_idxs:
            gridpass_dict={}
            for key_idx,key in enumerate(self._keys):
                gridpass_dict[key]=self._search_space[key][idxs[key_idx]]
            gridpass_f(self,**gridpass_dict)
            self.iter_count+=1
        self.reset()

# operator utilities
class ConvOps(nn.Module):
    def __init__(self,kernels) -> None:
        super().__init__()

        self.kernels=nn.parameter.Parameter(kernels, requires_grad=False)
        
        self.real=None
        self.DFT_kernels=None
        self.dft_shape=None
        self.x_shape=None

    def set_params(self,x):

        if (self.DFT_kernels is None) and (self.dft_shape is None) and (self.x_shape is None):
            self.x_shape=x.shape
            kernel_shape=self.kernels.shape[1:]
            self.dft_shape = tuple(np.array(kernel_shape) + np.array(x.shape[-len(kernel_shape):]) -1)
            self.DFT_kernels = torch.fft.fftn(self.kernels, s=self.dft_shape) 
            self.real = torch.all(torch.isreal(x))


    def _cast2real(self,x):
        if self.real:
            return torch.real(x)
        else:
            return x
    
    def _truncate(self,y):
        
        if len(self.dft_shape) == 3:
            return y[...,0:self.x_shape[-3],0:self.x_shape[-2],0:self.x_shape[-1]]
        elif len(self.dft_shape)== 2:
            return y[...,0:self.x_shape[-2],0:self.x_shape[-1]]
        elif len(self.dft_shape)== 1:
            return y[...,0:self.x_shape[-1]]
        else:
            return None


    def A(self,DFTx):
        M = self.DFT_kernels*DFTx
        m = torch.fft.ifftn(M,s=self.dft_shape)
        out = self._cast2real(m)
        return out

    def AH(self,DFTy):
        M = torch.conj(self.DFT_kernels)*DFTy
        m = torch.fft.ifftn(M,s=self.dft_shape)
        m = self._truncate(m)
        m = self._cast2real(m)
        out = torch.sum(m,dim=-len(self.kernels.shape),keepdim=True)
        return out
    
    def AHA(self,DFTx):
        M = (torch.abs(self.DFT_kernels)**2)*DFTx
        m = torch.fft.ifftn(M,s=self.dft_shape)
        m = self._truncate(m)
        m = self._cast2real(m)
        out = torch.sum(m,dim=-len(self.kernels.shape),keepdim=True)
        return out 

    def forward(self,signal,op='A',reg=0):
        
        DFTin=torch.fft.fftn(signal,s=self.dft_shape)

        if op=='A':
            return self.A(DFTx=DFTin)
        elif op=='AH':
            return self.AH(DFTy=DFTin)
        elif op=='AHA':
            return self.AHA(DFTx=DFTin)

def wrap_phase(x,op):
    xm=torch.abs(x)
    xph=torch.angle(x)
    return torch.polar(op(xm),xph)

def soft_shrinkage(prox_point,shrink):
    shrinkage_condition=(torch.abs(prox_point)-shrink)>0
    return torch.where(shrinkage_condition, prox_point-shrink,0)

def proj_rball(prox_point, r, dims_exclude=[0]):
    dist = l2_norm(prox_point,dims_exclude=dims_exclude)
    return torch.where(dist>r,r*(prox_point/dist),prox_point)

def proj_Rplus(prox_point):
    return torch.relu(torch.real(prox_point))

class prox_TV3D(nn.Module):
    """
    https://blog.allardhendriksen.nl/cwi-ci-group/chambolle_pock_using_tomosipo/
    """
    def __init__(self, max_iter=5,L=None,show_operator_norm_estimate=False) -> None:
        super().__init__()

        d = torch.zeros(size=(3,2,2,2),requires_grad=False)
  
        d[:,0,0,0]=1
        d[0,1,0,0]=-1
        d[1,0,1,0]=-1
        d[2,0,0,1]=-1

        self.grad_op = ConvOps(kernels=d)
        self.max_iter=max_iter
        self.show_operator_norm_estimate=show_operator_norm_estimate
        self.L=L
    
    def _set_operator_norm(self,prox_point,nof_iter=100):
        if self.L is None:
            x = torch.randn(prox_point.shape[-4:], device=prox_point.device)
            operator_norm_estimate = 0.0

            for i in range(nof_iter):
                y_A = x
                y_TV = self.grad_op(x)
                x_new = y_A + self.grad_op(y_TV,'AH')
                
                operator_norm_estimate = l2_norm(x_new) / l2_norm(x)
                x = x_new / l2_norm(x_new)

            self.L = operator_norm_estimate.item()
            print('Operator norm estimate is ',self.L) if self.show_operator_norm_estimate else None
         

    def _F_update(self,p_tilda):
        m = torch.sqrt(torch.sum(torch.abs(p_tilda)**2,dim=-4,keepdim=True))
        return p_tilda*torch.clamp(1/m,None,1)  
        
    def _G_update(self,u_tilda,tau,lmda,g):
        # g is the prox point
        return (u_tilda+tau*lmda*g)/(1+tau*lmda)
        
    

    def forward(self,prox_point,lmda):
        self.grad_op.set_params(prox_point)
        self._set_operator_norm(prox_point)


        y=prox_point
        u=prox_point
        u_dual=u
        p=0

        q = torch.zeros_like(self.grad_op(u))

        tau=0.01
        sigma=1/(tau*self.L**2)

        for i in range(self.max_iter):
            p = (p + sigma*(u_dual-y))/(1+sigma)

            g = self.grad_op(u_dual,'A')
            z = q + sigma*g
            m = torch.sqrt(torch.sum(torch.abs(z)**2,dim=-4,keepdim=True))
            q = z*torch.clamp(lmda/m,None,1)  

            u_prev=u
            u=u_prev-tau*(p+self.grad_op(q,'AH'))
            u_dual = u + tau*(u-u_prev)
        
        return u

# Some Cost Functions
class TVcost(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        d = torch.zeros(size=(3,2,2,2),requires_grad=False)
  
        d[:,0,0,0]=1
        d[0,1,0,0]=-1
        d[1,0,1,0]=-1
        d[2,0,0,1]=-1

        self.grad_op = ConvOps(kernels=d)
        
    def forward(self,x):
        self.grad_op.set_params(x)
        Dx=self.grad_op(torch.abs(x))
        magDx=torch.sqrt(torch.sum(torch.abs(Dx)**2,dim=-4,keepdim=True))
        return torch.sum(torch.abs(magDx))/magDx.shape[0]

class DFcost(nn.Module):
    def __init__(self,forward_model) -> None:
        super().__init__()
        self.A=forward_model
    def forward(self,x,y):
        return torch.mean(l2_norm(y-self.A(x)))

def relative_difference(x,x_den):
    return l2_norm(x-x_den)/l2_norm(x_den)

# ADMM Variants
class PNP_CSALSA(nn.Module):
    def __init__(self, forward_model, denoiser, data_fidelity, iter_recorder, zero_phase=False) -> nn.Module:
    
        super().__init__()
        
        self.A = forward_model
        self.denoiser = denoiser
        self.data_fidelity = data_fidelity
        self.zero_phase = zero_phase
        self.iter_recorder=iter_recorder
        
        self._reset_params()

    def _cast(self,signal):
        if self.zero_phase:
            return proj_Rplus(signal).to(dtype=torch.complex128)
        else:
            return signal

    def x_update(self,x,v1,d1,v2,d2,K):
        return self._cast(self.data_fidelity(x,v1,d1,v2,d2,K))

    def v1_update(self,noise_var,x,d1):
        prox_point=torch.abs(x-d1)
        ph=torch.angle(x-d1)
        mg=self.denoiser(prox_point,noise_var)
        return self._cast(torch.polar(mg,ph))
        
    def v2_update(self,Ax,d2,y,eps):
        #self.Ax=self.A(x)
        return y + proj_rball((Ax-d2-y),r=eps)

    def _reset_params(self):
        self.x_prev = None
        self.x = None
        self.x_ref = None
        self.Ax=None

        self.y = None
        self.terminated_iter=0
        self.eps=None
        self.noise_var=None
        self.K=None
        self.delta_n=None
        self.delta_K=None

        
        self.th=None
        self.max_iter=None
        self.x_ref=None

    def get_estimate(self):
        return torch.abs(self.x/maxmag3D(self.x))
    
    def forward(self, y, eps, noise_var, K, delta_n=1,delta_K=1, th=5e-4, max_iter=10000, tqdm_active=False,x_ref=None,x=None,asynchronous=True):
        
        # reset the previously registered params:
        self._reset_params()
        
        # register params:
        self.y=y
        self.eps=eps
        self.noise_var=noise_var
        self.K=K
        self.delta_n=delta_n
        self.delta_K=delta_K
        

        self.th=th
        self.max_iter=max_iter
        self.x_ref=x_ref

        # register x_(k)
        if x is None:
            # x_0 is set as (A'y)_+/max_voxel_value(|A'y|) 
            self.x = self._cast(self.A(y,adjoint=True))
            self.x = self.x/maxmag3D(self.x)
        else:
            self.x = x

        # register x_(k-1)
        self.x_prev = self.x.detach().clone()
        self.Ax = self.A(self.x)
        
        # dual variables
        v1=self.x+0
        v2=self.Ax+0
        
        # residuals
        d1=self.x-v1
        d2=self.Ax-v2
        
        if self.x.shape[0]==1:
            asynchronous=False

        # Start C-SALSA iters
        self.iter_recorder() # record before the first iteration
        if not asynchronous:
            for i in tqdm(range(self.max_iter),disable=not tqdm_active):
            
                # update x
                self.x = self.x_update(self.x_prev,v1,d1,v2,d2,self.K)
                
                # update v1
                v1=self.v1_update(noise_var=self.noise_var,x=self.x,d1=d1)
                
                # update v2
                self.Ax=self.A(self.x)
                v2=self.v2_update(self.Ax,d2,y,eps)

                # update d1 and d2
                d1=d1-(self.x-v1)
                d2=d2-(self.Ax-v2)
            

                self.iter_recorder() # record 
                # teriminate if improvement stops
                if i>0 and torch.all(relative_difference(x=torch.abs(self.x),x_den=torch.abs(self.x_prev)) < th):
                    break
                
                # update penalty parameters (or in this case noise variance)
                self.noise_var /= delta_n
                self.K *= delta_K
                
                # hold x 
                self.x_prev = self.x
        else:
            cont_opt = torch.ones(size=(self.x.shape[0],1,1,1,1), dtype=torch.bool,device=self.x.device).squeeze()
            for i in tqdm(range(self.max_iter),disable=not tqdm_active):
                
                # update x
                self.x[cont_opt,...] = self.x_update(self.x_prev[cont_opt,...],v1[cont_opt,...],d1[cont_opt,...],v2[cont_opt,...],d2[cont_opt,...],self.K)

                # update v1
                v1[cont_opt,...]=self.v1_update(noise_var=self.noise_var,x=self.x[cont_opt,...],d1=d1[cont_opt,...])

                # update v2
                self.Ax=self.A(self.x)
                v2[cont_opt,...]=self.v2_update(self.Ax[cont_opt,...],d2[cont_opt,...],y[cont_opt,...],eps[cont_opt,...])

                # update d1 and d2
                d1[cont_opt,...]=d1[cont_opt,...]-(self.x[cont_opt,...]-v1[cont_opt,...])
                d2[cont_opt,...]=d2[cont_opt,...]-(self.Ax[cont_opt,...]-v2[cont_opt,...])
            

                self.iter_recorder() # record 
                # teriminate if improvement stops
                rd=relative_difference(x=torch.abs(self.x),x_den=torch.abs(self.x_prev)).squeeze()
                if i>0 :
                    cont_opt = rd > th
                    if torch.all( torch.logical_not(cont_opt) ):
                        break
                
                # update penalty parameters (or in this case noise variance)
                self.noise_var /= delta_n
                self.K *= delta_K
                
                # hold x 
                self.x_prev = self.x.detach().clone()

        self.terminated_iter=i
        
class TVCG_CSALSA(PNP_CSALSA):
    def __init__(self, forward_model, chambolle_iter=5, cg_max_iter=5, cg_th=1e-5, zero_phase=False,record=False) -> nn.Module:

        denoiser = prox_TV3D(max_iter=chambolle_iter)
        self.cg_solver = cg3D 
        self.cg_max_iter=cg_max_iter
        self.cg_th=cg_th

        self.tvcost_history=[]
        self.dfcost_history=[]
        self.psnr_history=[]
        self.relative_difference_history=[]
        
        super().__init__(forward_model, denoiser, data_fidelity=self._data_fidelity, iter_recorder=self._recorder, zero_phase=zero_phase)
                        
        self.TVcost=TVcost()
        self.psnr = psnr3D
        self.record=record


    def _recorder(self):
        if self.record:
            self.tvcost_history.append(self.TVcost(self.x).to('cpu').item())
            self.dfcost_history.append(torch.mean(l2_norm(self.y-self.Ax)).to('cpu').item())
            self.psnr_history.append(torch.mean(self.psnr(torch.abs(self.x_ref),self.get_estimate())).to('cpu').item())            
            self.relative_difference_history.append(torch.mean(relative_difference(x=torch.abs(self.x),x_den=torch.abs(self.x_prev))).to('cpu').item())

    def clear_history(self):
        self.tvcost_history=[]
        self.dfcost_history=[]
        self.psnr_history=[]
        self.relative_difference_history=[]

    def _Q(self,x,K):
        Qx = K*x+self.A(self.A(x),adjoint=True)
        return Qx
        
    def _b(self,v1,d1,v2,d2,K):
        b=K*(v1+d1)+self.A(v2+d2,adjoint=True)
        return b
    
    def _data_fidelity(self,x,v1,d1,v2,d2,K):
        _Q=lambda xin: self._Q(xin,K)
        return self.cg_solver(Q=_Q,b=self._b(v1,d1,v2,d2,K),x=x,max_iter=self.cg_max_iter,th=self.cg_th,tqdm_active=False)

class l1CG_CSALSA(PNP_CSALSA):
    def __init__(self, forward_model, cg_max_iter=5, cg_th=1e-5, zero_phase=False,record=False) -> nn.Module:
        denoiser = soft_shrinkage
        self.cg_solver = cg3D 
        self.cg_max_iter=cg_max_iter
        self.cg_th=cg_th
        
        super().__init__(forward_model, denoiser, data_fidelity= self._data_fidelity, iter_recorder=self._recorder, zero_phase=zero_phase)
    
        self.l1cost=lambda x: torch.sum(torch.abs(x),dim=(-3,-2,-1))
        self.psnr = psnr3D
        self.record=record

        self.clear_history()

    def _recorder(self):
        if self.record:
            self.l1cost_history.append(torch.mean(self.l1cost(self.x)).to('cpu').item())
            self.dfcost_history.append(torch.mean(l2_norm(self.y-self.Ax)).to('cpu').item())
            self.psnr_history.append(torch.mean(self.psnr(torch.abs(self.x_ref),self.get_estimate())).to('cpu').item())            
            self.relative_difference_history.append(torch.mean(relative_difference(x=torch.abs(self.x),x_den=torch.abs(self.x_prev))).to('cpu').item())

    def clear_history(self):
        self.l1cost_history=[]
        self.dfcost_history=[]
        self.psnr_history=[]
        self.relative_difference_history=[]

    def _Q(self,x,K):
        Qx = K*x+self.A(self.A(x),adjoint=True)
        return Qx
        
    def _b(self,v1,d1,v2,d2,K):
        b=K*(v1+d1)+self.A(v2+d2,adjoint=True)
        return b
    
    def _data_fidelity(self,x,v1,d1,v2,d2,K):
        _Q=lambda xin: self._Q(xin,K)
        return self.cg_solver(Q=_Q,b=self._b(v1,d1,v2,d2,K),x=x,max_iter=self.cg_max_iter,th=self.cg_th,tqdm_active=False)

class NNCG_CSALSA(PNP_CSALSA):
    def __init__(self, forward_model, denoiser_net, blind=False, noise_level_map='std', cg_max_iter=5, cg_th=1e-5, zero_phase=False, record=False) -> nn.Module:
        
        self.blind = blind
        self.cg_solver = cg3D 
        self.cg_max_iter = cg_max_iter
        self.cg_th = cg_th
        self.noise_level_map=noise_level_map

        self.dfcost_history=[]
        self.psnr_history=[]
        self.relative_difference_history=[]

        denoiser = None
        if self.blind:
            denoiser=self._denoise_blind
        else:
            if self.noise_level_map=='std':
                denoiser = self._denoise_nlm_std
            elif self.noise_level_map=='var':
                denoiser=self._denoise_nlm_var

        super().__init__(forward_model=forward_model, denoiser=denoiser, 
                        data_fidelity=self._data_fidelity, zero_phase=zero_phase,
                        iter_recorder=self._recorder)

        self.denoiser_net = denoiser_net                
        self.DFcost=DFcost(forward_model=forward_model)
        self.psnr=psnr3D

        self.record=record

    def _denoise_nlm_std(self,x,noise_var):
        with torch.no_grad():
            self.denoiser_net.eval()
            x=x.to(dtype=torch.float)    
            x_vr = torch.concat((x,torch.ones_like(x)*((noise_var)**(1/2))),dim=-4)

            return self.denoiser_net.forward(x_vr).to(torch.double)
    
    def _denoise_nlm_var(self,x,noise_var):
        with torch.no_grad():
            self.denoiser_net.eval()
            x=x.to(dtype=torch.float)    
            x_vr = torch.concat((x,torch.ones_like(x)*(noise_var)),dim=-4)
            
            return self.denoiser_net.forward(x_vr).to(torch.double)
        
    def _denoise_blind(self,x,noise_var):
        with torch.no_grad():
            self.denoiser_net.eval()
            x=x.to(dtype=torch.float)
            return self.denoiser_net.forward(x).to(torch.double)

    def _recorder(self):
        if self.record:
            self.dfcost_history.append(self.DFcost(self.x,self.y).to('cpu').item())
            self.psnr_history.append(torch.mean(self.psnr(torch.abs(self.x_ref),self.get_estimate())).to('cpu').item())
            self.relative_difference_history.append(torch.mean(relative_difference(x=torch.abs(self.x),x_den=torch.abs(self.x_prev))).to('cpu').item())

    def clear_history(self):
        self.dfcost_history=[]
        self.psnr_history=[]
        self.relative_difference_history=[]
    
    def _Q(self,x,K):
        Qx = K*x+self.A(self.A(x),adjoint=True)
        return Qx
        
    def _b(self,v1,d1,v2,d2,K):
        b=K*(v1+d1)+self.A(v2+d2,adjoint=True)
        return b
    
    def _data_fidelity(self,x,v1,d1,v2,d2,K):
        _Q=lambda xin: self._Q(xin,K)
        return self.cg_solver(Q=_Q,b=self._b(v1,d1,v2,d2,K),x=x,max_iter=self.cg_max_iter,th=self.cg_th,tqdm_active=False)    

class BackProjection(nn.Module):
    def __init__(self,radar_system_params) -> None:
        super().__init__()

        self.freqs = radar_system_params.freqs
        self.tx_positions = radar_system_params.tx_positions
        self.rx_positions = radar_system_params.rx_positions
        self.x_coords = radar_system_params.x_coords
        self.y_coords = radar_system_params.y_coords
        self.z_coords = radar_system_params.z_coords

        self.ks = 2*torch.pi*self.freqs/C_SOL
        self.A=nn.Parameter(self._getBPmat())
        self.x=None

    def _BPkernel(self,k, R_tx, R_rx):
        return torch.exp(1j*k*(R_rx+R_tx))
    def _getBPmat(self):
        A=torch.zeros(size=(self.ks.size()[0],
                            self.tx_positions.shape[0],
                            self.rx_positions.shape[0],
                            self.x_coords.size()[0],
                            self.y_coords.size()[0],
                            self.z_coords.size()[0]),
                            dtype=torch.complex128)

        scene_coords = tensor.mgrid(self.x_coords, self.y_coords, self.z_coords)

        for k_idx,k in enumerate(self.ks):
            for tx_idx,tx_pos in enumerate(self.tx_positions):
                for rx_idx,rx_pos in enumerate(self.rx_positions):
                    R_rx = tensor.mgrid_l2(scene_coords-rx_pos)
                    R_tx = tensor.mgrid_l2(scene_coords-tx_pos)
                    A[k_idx,tx_idx,rx_idx,:,:,:] = self._BPkernel(k=k,R_tx=R_tx,R_rx=R_rx)
                    
        return A

    def forward(self,y):
        self.x=torch.einsum('KTRxyz,NCKTR->NCxyz',self.A,y)

    def get_estimate(self):
        return torch.abs(self.x/maxmag3D(self.x))
    
class KirchhoffMigration(nn.Module):
    def __init__(self,radar_system_params,compansate_limited_aparture=True,compute_by_forward_loop=False) -> None:
        super().__init__()
        self.compansate=compansate_limited_aparture
        self.compute_by_forward_loop=compute_by_forward_loop

        self.freqs = radar_system_params.freqs
        self.tx_positions = radar_system_params.tx_positions
        self.rx_positions = radar_system_params.rx_positions
        self.x_coords = radar_system_params.x_coords
        self.y_coords = radar_system_params.y_coords
        self.z_coords = radar_system_params.z_coords

        self.ks = 2*torch.pi*self.freqs/C_SOL
        self.A=nn.Parameter(self._getBackwardMIMOmat(self.compansate)) if not self.compute_by_forward_loop else nn.Parameter(torch.empty(1))
        self.x=None

    def _getBackwardMIMOmat(self,compansate):
        A=torch.zeros(size=(self.ks.size()[0],
                            self.tx_positions.shape[0],
                            self.rx_positions.shape[0],
                            self.x_coords.size()[0],
                            self.y_coords.size()[0],
                            self.z_coords.size()[0]),
                            dtype=torch.complex128)

        scene_coords = tensor.mgrid(self.x_coords, self.y_coords, self.z_coords)

        for k_idx,k in enumerate(self.ks):
            for tx_idx,tx_pos in enumerate(self.tx_positions):
                for rx_idx,rx_pos in enumerate(self.rx_positions):
                    R_rx = tensor.mgrid_l2(scene_coords-rx_pos)
                    R_tx = tensor.mgrid_l2(scene_coords-tx_pos)

                    z = scene_coords[:,:,:,2]
                    z_rx = rx_pos[2]
                    z_tx = tx_pos[2]                    

                    dRrx_dz = (z-z_rx)/R_rx
                    dRtx_dz = (z-z_tx)/R_tx

                    A[k_idx,tx_idx,rx_idx,:,:,:] = 4*(dRrx_dz)*(dRtx_dz)* ( R_rx*R_tx*((1j*k)**2) + (R_rx+R_tx)*(1j*k) + 1)*torch.exp(1j*k*(R_rx+R_tx))
                    if not compansate:
                        A[k_idx,tx_idx,rx_idx,:,:,:]=A[k_idx,tx_idx,rx_idx,:,:,:]/(4*R_rx*R_tx)

        return A

    def _forward_loop(self,y,tqdm_active=True):
        device=self.A.device
        compansate=self.compansate
        scene_coords = tensor.mgrid(self.x_coords, self.y_coords, self.z_coords).to(device)
        x_est = torch.zeros(size=(y.shape[0],1,scene_coords.shape[0],scene_coords.shape[1],scene_coords.shape[2]),dtype=torch.complex128,device=device)

        y=y.to(device)
        for k_idx,k in tqdm(enumerate(self.ks.to(device)),disable=not tqdm_active):
            for tx_idx,tx_pos in enumerate(self.tx_positions.to(device)):
                for rx_idx,rx_pos in enumerate(self.rx_positions.to(device)):
                    R_rx = tensor.mgrid_l2(scene_coords-rx_pos)
                    R_tx = tensor.mgrid_l2(scene_coords-tx_pos)

                    z = scene_coords[:,:,:,2]
                    z_rx = rx_pos[2]
                    z_tx = tx_pos[2]                    

                    dRrx_dz = (z-z_rx)/R_rx
                    dRtx_dz = (z-z_tx)/R_tx

                    a_k_idx_tx_idx_rx_idx = 4*(dRrx_dz)*(dRtx_dz)* ( R_rx*R_tx*((1j*k)**2) + (R_rx+R_tx)*(1j*k) + 1)*torch.exp(1j*k*(R_rx+R_tx))
                    del_x = a_k_idx_tx_idx_rx_idx*y[...,k_idx,tx_idx,rx_idx]

                    if not compansate:
                        del_x/=(4*R_rx*R_tx)
                    x_est+=del_x
        return x_est

    def forward(self,y,tqdm_active=False):
        if self.compute_by_forward_loop:
            self.x=self._forward_loop(y,tqdm_active)
        else:
            self.x=torch.einsum('KTRxyz,NCKTR->NCxyz',self.A,y)

    def get_estimate(self):
        return torch.abs(self.x/maxmag3D(self.x))
 
