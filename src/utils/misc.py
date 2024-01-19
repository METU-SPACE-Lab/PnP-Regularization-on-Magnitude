import numpy as np
import torch
from tqdm import tqdm
import gc

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import src.utils.tensor as tensor

import matplotlib
import matplotlib.pyplot as plt


def t2np_mgrid(t_mgrid):
    np_mgrid=np.array(t_mgrid)
    nof_dims=np_mgrid.shape[-1]
    permuted_axes=tuple( [nof_dims] + [i for i in range(0,nof_dims)])

    return np.transpose(np_mgrid,axes=permuted_axes)

def quickplot_density(mgrid, values, surface_count=10,opacity=0.2,colorbar_nticks=5, isomin=None,isomax=None, showscale=True):
    
    if(type(mgrid)==torch.Tensor):
        mgrid = t2np_mgrid(mgrid)
        X, Y, Z = mgrid
        
    if len(values.shape)==5:
        values=np.squeeze(values)

    values=np.array(values)
    if isomin is None:
        isomin=np.min(values)
    if isomax is None:
        isomax=np.max(values)

    if len(values.shape)==3:        

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            isomin=isomin,
            isomax=isomax,
            colorbar_nticks=colorbar_nticks,
            showscale=showscale,
            value=values.flatten(),
            opacity=opacity, 
            surface_count=surface_count,
            ))                    
        fig.update_layout(scene_xaxis_showticklabels=True,
                        scene_yaxis_showticklabels=True,
                        scene_zaxis_showticklabels=True)
        return fig

            
    if len(values.shape)==4:
        
               
        N = values.shape[0]
        
        fig = make_subplots(
        rows=1, cols=N,
        specs=[[{'type': 'volume'} for i in range(N)]])

        for i in range(N):
            fig.add_trace(
                go.Volume(
                    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                    value=values[i,:,:,:].flatten(),
                    opacity=opacity,
                    isomin=isomin,
                    isomax=isomax,
                    colorbar_nticks=colorbar_nticks,
                    showscale=showscale,
                    surface_count=surface_count),row=1,col=i+1)
        return fig

def write(str):
    tqdm.write(str)

def sceneshow(x,radar_system_params,isomin=None,isomax=None,width=None,showscale=True,zoom_out=0):
    if width is None:
        width=x.shape[0]*500
       
    mgrid = tensor.mgrid(radar_system_params.x_coords,radar_system_params.y_coords,radar_system_params.z_coords)
    show = torch.abs(x)
    fig = quickplot_density(mgrid,values=show.to(device='cpu'),surface_count=15,opacity=0.2,colorbar_nticks=10,isomin=isomin,isomax=isomax,showscale=showscale)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_scenes(camera=dict(eye=dict(x=1.4+zoom_out, y=1.4+zoom_out, z=1.4+zoom_out)))
    fig.update_layout(width=int(width)+50,height=500)
    return fig

def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()

def super_clear_cuda(N):
    for n in range(N):
        clear_cuda()
clear_cuda()

class Scene():
    def __init__(self) -> None:    
        pass

    def _sceneshow_yarovoy(self,x,radar_system_params,projection='perspective'):
        import copy
        if type(x) is not torch.Tensor:
            x=torch.concat(x,dim=0)

        radar_system_params_yarovoy = copy.copy(radar_system_params)
        radar_system_params_yarovoy.x_coords=radar_system_params.z_coords
        radar_system_params_yarovoy.z_coords=radar_system_params.x_coords
        radar_system_params_yarovoy.y_coords=radar_system_params.y_coords

        # fig = sceneshow(x=x,radar_system_params=radar_system_params_yarovoy,showscale=False)
        # fig = sceneshow(x=torch.transpose(x,dim0=-3,dim1=-1),radar_system_params=radar_system_params_yarovoy,showscale=False)
        fig = sceneshow(x=torch.flip(torch.transpose(x,dim0=-3,dim1=-1),dims=(-2,-1)),radar_system_params=radar_system_params_yarovoy,showscale=False,isomax=1,isomin=0)
        
        d=.85

        if projection == 'orthographic':
            fig.update_scenes(camera=dict(eye=dict(x=-2.5*d, y=-1.5*d, z=1.5*d),projection=dict(type=projection)))
        elif projection == 'perspective':
            fig.update_scenes(camera=dict(eye=dict(x=-2.5*d, y=-1.5*d, z=1.5*d),projection=dict(type=projection)))    
        fig.update_layout(width=600*(x.shape[0]), height=500, margin=dict(t=30, r=50, l=50, b=0))
        return fig

    def show_experimental(self,x,radar_system_params):

        fig=self._sceneshow_yarovoy(x=x,radar_system_params=radar_system_params)
        fig.update_layout(margin=dict(t=0, r=0, l=0, b=0), 
            font=dict(
                size=15,
                color="black"),
            scene_zaxis = dict(
            tickmode = 'array',
            tickvals = [-0.13,0,0.15],
            ticktext =['-0.15','0','0.15'],
            title='x (m)',
            ticks="inside",    
            tickangle=0,

            ),
            scene_yaxis = dict(
            tickmode = 'array',
            tickvals = [-0.15,0,0.13],
            ticktext =['-0.15','0','0.15'],
            title='y (m)',
            tickangle=0,
            ticks="inside",    
            ),
            scene_xaxis = dict(
            tickmode = 'array',
            tickvals = [0.35,0.5,0.65],
            ticktext = ['0.35','0.50','0.65'],
            title='z (m)',
            tickangle=0,
            ticks="inside",    
            ),
            
            height=500,
            width=500     
        )
        fig.update_layout(title=None)

        return fig

    def show_simulated(self,x,radar_system_params, mark_axes=True, show_colorbar=False):


        fig = sceneshow(x=x,isomin=0,isomax=1,radar_system_params=radar_system_params,showscale=show_colorbar)
        fig.update_scenes(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5),projection=dict(type='perspective')))
        fig.update_scenes(yaxis=dict(title='y (m)'),zaxis=dict(title='z (m)'),xaxis=dict(title='x (m)'))
        if not mark_axes:
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), 
            font=dict(
                size=15,
                color="black"),
            scene_xaxis = dict(
            tickmode = 'array',
            tickvals = [],
            ticktext = [],
            title='',
            tickangle=0,

            ),
            scene_yaxis = dict(
            tickmode = 'array',
            tickvals = [],
            ticktext = [],
            title='',
            tickangle=0,

            ),
            scene_zaxis = dict(
            tickmode = 'array',
            tickvals = [],
            ticktext = [],
            title='',
            tickangle=0,
            ),
            height=500,
            width=500
            )
        else:
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), 
            font=dict(
                size=15,
                color="black"),
            scene_xaxis = dict(
            tickmode = 'array',
            tickvals = [-0.15,0,0.13],
            ticktext = ['-0.15','0','0.15'],
            title='x (m)',
            ticks="inside",    
            tickangle=0,

            ),
            scene_yaxis = dict(
            tickmode = 'array',
            tickvals = [-0.15,0,0.13],
            ticktext = ['-0.15','0','0.15'],
            title='y (m)',
            tickangle=0,
            ticks="inside",    
            ),
            scene_zaxis = dict(
            tickmode = 'array',
            tickvals = [0.37,0.5,0.65],
            ticktext = ['0.35','0.5','0.65'],
            title='z (m)',
            tickangle=0,
            ticks="",    

            ),
            height=500,
            width=700,       
            )
        return fig