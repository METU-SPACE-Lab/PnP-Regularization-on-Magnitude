FORWARD_MATRIX_PATH = 'data/temp/A.pkl'
DATA_ROOT = 'data/datasets'

DATASET_PATHS=\
{
    'IrfanC':
    {        
        'train': '/'.join((DATA_ROOT,'IrfanC/train.pt')),
        'test': '/'.join((DATA_ROOT,'IrfanC/test.pt')),
        'val': '/'.join((DATA_ROOT,'IrfanC/val.pt')),
        'search':'/'.join((DATA_ROOT,'IrfanC/search.pt'))
    },
    'Yarovoy':
    {
        'measurements' : DATA_ROOT+'/experimental/Yarovoy/measurements.pt', #Experimental Data from '3-D Short-Range Imaging With Irregular MIMO Arrays Using NUFFT-Based Range Migration Algorithm'
        'tx_positions' :  DATA_ROOT+'/experimental/Yarovoy/tx_positions.pt',
        'rx_positions' : DATA_ROOT+'/experimental/Yarovoy/rx_positions.pt',
        'freqs': DATA_ROOT+'/experimental/Yarovoy/freqs.pt'
    }
}


RADAR_SYSTEM_CONFIGS=\
{
    15:{\
        'FREQ_MIN' : 4e9,
        'FREQ_MAX' : 16e9,
        'FREQ_STEPS' : 15,

        'N_tx':12,
        'N_rx':13,
        'ARRAY_WIDTH' : 0.3,

        'X_MIN':-0.15,
        'X_MAX':0.15,
        'N_X':25,
        
        'Y_MIN':-0.15,
        'Y_MAX':0.15,
        'N_Y':25,
        
        'Z_MIN':0.35,
        'Z_MAX':0.65,
        'N_Z':49
        },
    10:{\
        'FREQ_MIN' : 4e9,
        'FREQ_MAX' : 16e9,
        'FREQ_STEPS' : 10,

        'N_tx':12,
        'N_rx':13,
        'ARRAY_WIDTH' : 0.3,

        'X_MIN':-0.15,
        'X_MAX':0.15,
        'N_X':25,
        
        'Y_MIN':-0.15,
        'Y_MAX':0.15,
        'N_Y':25,
        
        'Z_MIN':0.35,
        'Z_MAX':0.65,
        'N_Z':49
        },        
    20:{\
        'FREQ_MIN' : 4e9,
        'FREQ_MAX' : 16e9,
        'FREQ_STEPS' : 20,

        'N_tx':12,
        'N_rx':13,
        'ARRAY_WIDTH' : 0.3,

        'X_MIN':-0.15,
        'X_MAX':0.15,
        'N_X':25,
        
        'Y_MIN':-0.15,
        'Y_MAX':0.15,
        'N_Y':25,
        
        'Z_MIN':0.35,
        'Z_MAX':0.65,
        'N_Z':49
        },
    30:{\
        'FREQ_MIN' : 4e9,
        'FREQ_MAX' : 16e9,
        'FREQ_STEPS' : 30,

        'N_tx':12,
        'N_rx':13,
        'ARRAY_WIDTH' : 0.3,

        'X_MIN':-0.15,
        'X_MAX':0.15,
        'N_X':25,
        
        'Y_MIN':-0.15,
        'Y_MAX':0.15,
        'N_Y':25,
        
        'Z_MIN':0.35,
        'Z_MAX':0.65,
        'N_Z':49
        },
    40:{\
        'FREQ_MIN' : 4e9,
        'FREQ_MAX' : 16e9,
        'FREQ_STEPS' : 40,

        'N_tx':12,
        'N_rx':13,
        'ARRAY_WIDTH' : 0.3,

        'X_MIN':-0.15,
        'X_MAX':0.15,
        'N_X':25,
        
        'Y_MIN':-0.15,
        'Y_MAX':0.15,
        'N_Y':25,
        
        'Z_MIN':0.35,
        'Z_MAX':0.65,
        'N_Z':49
        },
    5:{\
        'FREQ_MIN' : 4e9,
        'FREQ_MAX' : 16e9,
        'FREQ_STEPS' : 5,

        'N_tx':12,
        'N_rx':13,
        'ARRAY_WIDTH' : 0.3,

        'X_MIN':-0.15,
        'X_MAX':0.15,
        'N_X':25,
        
        'Y_MIN':-0.15,
        'Y_MAX':0.15,
        'N_Y':25,
        
        'Z_MIN':0.35,
        'Z_MAX':0.65,
        'N_Z':49
        },
}

import torch
import src.utils.tensor as tensor

class RSC2params():
    def __init__(self,Radar_System_Config) -> None:
        RSC=Radar_System_Config

        FREQ_MIN=RSC['FREQ_MIN']
        FREQ_MAX=RSC['FREQ_MAX']
        FREQ_STEPS=RSC['FREQ_STEPS']

        ARRAY_WIDTH=RSC['ARRAY_WIDTH']
        N_tx=RSC['N_tx']
        N_rx=RSC['N_rx']

        X_MIN=RSC['X_MIN']
        X_MAX=RSC['X_MAX']
        N_X=RSC['N_X']

        Y_MIN=RSC['Y_MIN']
        Y_MAX=RSC['Y_MAX']
        N_Y=RSC['N_Y']
        
        Z_MIN=RSC['Z_MIN']
        Z_MAX=RSC['Z_MAX']
        N_Z=RSC['N_Z']
        

        self.freqs = torch.linspace(start=FREQ_MIN,end=FREQ_MAX,steps=FREQ_STEPS)
        self.rx_positions=tensor.linspace(torch.tensor([-1,-1,0])*ARRAY_WIDTH/2,torch.tensor([+1,+1,0])*ARRAY_WIDTH/2,N_rx)
        self.tx_positions=tensor.linspace(torch.tensor([-1,+1,0])*ARRAY_WIDTH/2,torch.tensor([+1,-1,0])*ARRAY_WIDTH/2,N_tx)

        self.x_coords=torch.linspace(X_MIN,X_MAX,N_X)
        self.y_coords=torch.linspace(Y_MIN,Y_MAX,N_Y)
        self.z_coords=torch.linspace(Z_MIN,Z_MAX,N_Z)


