import copy
from copy import deepcopy 

import torch
from torch import nn

from scipy.constants import speed_of_light as C_SOL
from src.config import FORWARD_MATRIX_PATH
from src.utils import tensor,misc,data


def scattering_kernel(k, R_tx, R_rx):
    return torch.exp(-1j*k*(R_rx+R_tx))/(R_rx*R_tx*4*torch.pi)


class MVProd(nn.Module):
    """MVProd is a nn.Module that computes the measurements using Born approximation via matrix-vector product for a planar array.

    """
    def __init__(self, freqs, tx_positions, rx_positions,
                        x_coords, y_coords, z_coords, 
                        path, save = True): 
        super(MVProd, self).__init__()

        self.ks = 2*torch.pi*freqs/C_SOL
        self.tx_positions = tx_positions
        self.rx_positions = rx_positions
        
        self.x_coords=x_coords
        self.y_coords=y_coords
        self.z_coords=z_coords
        
        self.radar_dict=self.get_radar_dict()
                
        self.save=save
        self.A = nn.Parameter(self._load_A(path))
        self.A.requires_grad_(False)

        self._save_radar_dict(path) if self.save else None

    def _load_A(self,path):
        if data.path_exists(path):
            loaded_radar_dict=data.pickle_load(path)
            
            load_bool=True
            for key in list(self.radar_dict.keys()):
                if torch.equal(loaded_radar_dict[key],self.radar_dict[key]): 
                    load_bool = load_bool and True
                else:
                    load_bool = load_bool and False
            
            if load_bool:
                print('<--loading the radar config.')  
                self.save=False
                return loaded_radar_dict['A']
            else:
                print('...computing the forward matrix')
                return self._compute_A()
        else:
            print('...computing the forward matrix')
            return self._compute_A()
    
    def _save_radar_dict(self,path):
        print('-->saving the radar config.')
        
        save_dict=copy.deepcopy(self.radar_dict)
        save_dict['A']=self.A
        
        data.pickle_dump(data=save_dict,path=path) 
        
    def get_radar_dict(self):
        radar_dict={}
        radar_dict['tx_positions']=self.tx_positions
        radar_dict['rx_positions']=self.rx_positions
        radar_dict['ks']=self.ks
        radar_dict['x_coords']=self.x_coords
        radar_dict['y_coords']=self.y_coords
        radar_dict['z_coords']=self.z_coords
        
        return radar_dict

    def _compute_A(self):

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
                    A[k_idx,tx_idx,rx_idx,:,:,:] = scattering_kernel(k=k,R_tx=R_tx,R_rx=R_rx)
                    
        return A

    def _As(self,scene):
        """
        returns measurements (of shape (# freqs)(# transmitter)(# reciever)),
        for the scene tensors (of shape (# voxels. in x)(# voxels. in y)(# voxels. in z)). 
        If there are mutliple scene tensors (shape (# tensors)(# voxels. in x)(# voxels. in y)(# voxels. in z)),
        then returns mesurements for each tensor (shape (# tensors)(# freqs)(# transmitter)(# reciever)) 
        """
        if len(scene.shape)==3:
            return torch.einsum('ktrxyz,xyz->ktr',self.A,scene)
        if len(scene.shape)==4:
            return torch.einsum('ktrxyz,Nxyz->Nktr',self.A,scene)  
        if len(scene.shape)==5: 
            # The input and output of the Denoiser NN is NCxyz where C=1 
            return torch.einsum('ktrxyz,NCxyz->NCktr',self.A,scene)
          

    def _adjAy(self,meas):
        if len(meas.shape)==3:
            return torch.einsum('ktrxyz,ktr->xyz',torch.conj(self.A),meas)
        if len(meas.shape)==4:
            return torch.einsum('ktrxyz,Nktr->Nxyz',torch.conj(self.A),meas)
        if len(meas.shape)==5: 
            # This case is just to make the adjoint op. compatible with NN inputs and outputs. 
            # The input and output of the Denoiser NN is NCxyz where C=1 
            return torch.einsum('ktrxyz,NCktr->NCxyz', torch.conj(self.A),meas)

    def forward(self,input_signal,adjoint=False):

        if adjoint:
            return self._adjAy(meas=input_signal)
        else:
            return self._As(scene=input_signal)
        


class MVProd_iterK(MVProd):
    """MVProd_iterK is a nn.Module that computes the measurements using Born approximation via matrix-vector product for a planar array. Different than MVProd, MVProd_iterK computes the measurement for each frequency in batches thus saving memory. 

    """
    def __init__(self, freqs, tx_positions, rx_positions,
                        x_coords, y_coords, z_coords, 
                        path, save = True, f_batch_size=1): 

        assert torch.all(torch.linspace(freqs.min(),freqs.max(),freqs.size()[0],dtype=torch.float64) == freqs)
        
        self.n_freqs=freqs.size()[0]
        super().__init__(freqs=freqs[0:f_batch_size], tx_positions=tx_positions, rx_positions=rx_positions,
                        x_coords=x_coords, y_coords=y_coords, z_coords=z_coords, 
                        path=path, save = save)

        freq_period = freqs[-1]-freqs[-2]
        self.delta_k = (2*torch.pi*freq_period/C_SOL)*f_batch_size
        self.delta_phase = nn.Parameter(self._compute_delPhase())
        self.f_batch_size = f_batch_size
        self.f_iters = int(np.ceil(self.n_freqs/self.f_batch_size))

    def _compute_delPhase(self):
        delta_phase=torch.zeros(size=(1,
                            self.tx_positions.shape[0],
                            self.rx_positions.shape[0],
                            self.x_coords.size()[0],
                            self.y_coords.size()[0],
                            self.z_coords.size()[0]),
                            dtype=torch.complex128)

        scene_coords = tensor.mgrid(self.x_coords, self.y_coords, self.z_coords)

        for tx_idx,tx_pos in enumerate(self.tx_positions):
            for rx_idx,rx_pos in enumerate(self.rx_positions):
                R_rx = tensor.mgrid_l2(scene_coords-rx_pos)
                R_tx = tensor.mgrid_l2(scene_coords-tx_pos)
                delta_phase[0,tx_idx,rx_idx,:,:,:] = torch.exp(-1j*self.delta_k*(R_rx+R_tx))
                    
        return delta_phase


   
    def _As(self,scene):
        """
        returns measurements (of shape (# freqs)(# transmitter)(# reciever)),
        for the scene tensors (of shape (# voxels. in x)(# voxels. in y)(# voxels. in z)). 
        If there are mutliple scene tensors (shape (# tensors)(# voxels. in x)(# voxels. in y)(# voxels. in z)),
        then returns mesurements for each tensor (shape (# tensors)(# freqs)(# transmitter)(# reciever)) 
        """
        res=[]
        for i in range(self.f_iters):
            # The input and output of the Denoiser NN is NCxyz where C=1 
            f_slice = self.f_batch_size-max((i+1)*self.f_batch_size-self.n_freqs,0)
            res.append(torch.einsum('ktrxyz,NCxyz->NCktr',(self.delta_phase**i)*self.A[0:f_slice,...],scene))
        return torch.concat(res,dim=2)

    def _adjAy(self,meas):
        res=0
        for i in range(self.f_iters):
            mx=max((i+1)*self.f_batch_size-self.n_freqs,0)
            f_slice = self.f_batch_size- mx
            res+= torch.einsum('ktrxyz,NCktr->NCxyz', torch.conj((self.delta_phase**i)*self.A[0:f_slice,...]),meas[:,:,i*self.f_batch_size:(i+1)*self.f_batch_size-mx,:,:])
        return res
        