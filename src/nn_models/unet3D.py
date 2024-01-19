import torch
import torch.nn as nn
from src.nn_models.building_blocks import Tconv3D, CBNR3Dx2,Sconv3D
from src.utils.stats import maxmag3D

class denoiser(nn.Module):
    def __init__(self, start_ch= 32, nof_levels=3 , kernel_size = 3, E2D_extra=0, sig=False, res=False, nrm=False, blind=False, bias= True):
        super().__init__()

        self.start_ch=start_ch
        self.nof_levels=nof_levels
        self.kernel_size = kernel_size
        self.blind = blind 
        self.bias = bias

        self._handle_dim_mismatch_0 = nn.ConstantPad3d((0,1,0,0,0,0), 0)
        self._handle_dim_mismatch_1 = nn.ConstantPad3d((0,0,0,1,0,0), 0)
        self._handle_dim_mismatch_2 = nn.ConstantPad3d((0,0,0,0,0,1), 0)

        encoder_idx = list(range(0,nof_levels-1,1))
        decoder_idx = list(reversed(encoder_idx))
        
        E2D_module_list=nn.ModuleList(
            [CBNR3Dx2(in_channels = start_ch*(2**(nof_levels-2)),
                    out_channels = start_ch*(2**(nof_levels-1)), 
                    kernel_size = kernel_size, bias = bias)])
        
        for i in range(E2D_extra):
            E2D_module_list.append(
                CBNR3Dx2(in_channels = start_ch*(2**(nof_levels-1)),
                                out_channels = start_ch*(2**(nof_levels-1)), 
                                kernel_size = kernel_size, bias = bias))

        self.levels=nn.ModuleDict({
            'encoder':  nn.ModuleList(
                                [CBNR3Dx2(  in_channels=self.get_ioch(i,'enc')[0],
                                            out_channels=self.get_ioch(i,'enc')[1],
                                            kernel_size=kernel_size,
                                            bias=bias) for i in encoder_idx]), 

            'decoder':  nn.ModuleList(
                                [CBNR3Dx2(  in_channels=self.get_ioch(i,'dec')[0],
                                            out_channels=self.get_ioch(i,'dec')[1],
                                            kernel_size=kernel_size,
                                            bias=bias) for i in decoder_idx]), 

            'enc2dec':  nn.Sequential(*E2D_module_list),

            'u_sample': nn.ModuleList([nn.ConvTranspose3d( in_channels=self.get_ioch(i,'dec')[0],
                                            out_channels=self.get_ioch(i,'dec')[1],
                                            kernel_size=2,
                                            stride=2,
                                            bias=bias) for i in decoder_idx]), 
            
            'd_sample': nn.MaxPool3d(kernel_size=2,stride=2),
            
            'dec2out':  nn.Conv3d(in_channels=self.start_ch,out_channels=1,
                                 kernel_size=1, bias=bias, padding='same')

            })

        self.res = res
        self.sig = sig 
        self.nrm = nrm

        self.sigmoid=nn.Sigmoid()
        
    
    def get_ioch(self, level_idx, enc_dec='enc'):
        out_channels = self.start_ch*(2**(level_idx))
        
        in_channels = None
        if enc_dec=='dec':
            in_channels = out_channels*2
        elif enc_dec=='enc':
            if self.blind:
                in_channels = 1 if level_idx==0 else out_channels/2
            else:
                in_channels = 2 if level_idx==0 else out_channels/2

        return (int(in_channels),int(out_channels))
    
   
    def forward(self,xn_vr):
        
        x=xn_vr
        if self.nrm:
            x[:,0:1,...]=x[:,0:1,...]/maxmag3D(x[:,0:1,...])
        xn = x[:,0:1,...]
                
        latent_outputs=[]
        
        for i in range(self.nof_levels-1):
            z=self.levels['encoder'][i](x)
            latent_outputs.insert(0,z)
            x=self.levels['d_sample'](z)
        
        x=self.levels['enc2dec'](x)

        for i in range(self.nof_levels-1):
            x_up=self.levels['u_sample'][i](x)
            
            if x_up.shape[-1] != latent_outputs[i].shape[-1]:
                x_up= self._handle_dim_mismatch_0(x_up)
            if x_up.shape[-2] != latent_outputs[i].shape[-2]:
                x_up= self._handle_dim_mismatch_1(x_up)
            if x_up.shape[-3] != latent_outputs[i].shape[-3]:
                x_up= self._handle_dim_mismatch_2(x_up)

            z=torch.concat((x_up,latent_outputs[i]),dim=1)
            x=self.levels['decoder'][i](z)
        
        x=self.levels['dec2out'](x)

        if self.res:
            x=x+xn

        if self.sig:
            x=self.sigmoid(x)

        return x

