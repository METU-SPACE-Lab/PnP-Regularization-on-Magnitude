
# [Plug-and-Play Regularization on Magnitude with Deep Priors for 3D Near-Field MIMO Imaging](https://arxiv.org/abs/2312.16024)
[Okyanus Oral](https://ookyanus.github.io) and [Figen S. Oktem](https://blog.metu.edu.tr/figeno/).

This repository contains the official codes for the paper "[**Plug-and-Play Regularization on Magnitude with Deep Priors for 3D Near-Field MIMO Imaging**](https://arxiv.org/abs/2312.16024)". **(*to appear in IEEE Transactions on Computational Imaging*)**


![Experimental Reconstructions](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/images/ExperimentalResults-v2.png "Experimental Reconstructions")
<table>
<td> <a href="https://youtu.be/Q4pkmQCpx-U">Youtube: 3D Rotating views (Experimental Results)</a><br><img src="https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/images/Experimental Reconstructions.gif" width=390px />  </td>
<td> <a href="https://youtu.be/imN5wFll0hw">Youtube: 3D Rotating views (Simulation Results)</a><br><img src="https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/images/Simulated Reconstructions.gif" width=390px />  </td>
</table>

## FAQs: Simulated and Experimental Data
You can download the synthetically generated dataset from [here](https://drive.google.com/drive/folders/1sxosLDMB55ZEjkti-o2d7m3V59jCAe5o?usp=sharing). If you use this dataset, you should cite [1].

You should refer to (and cite) [2,3] for the experimental data.

Check [format-exp-data.ipynb](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/format-exp-data.ipynb) for the instructions on formatting the experimental data provided at [3].

**[1]** I. Manisali, O. Oral, and F. S. Oktem, ”Efficient physics-based learned reconstruction methods for real-time 3D near-field MIMO radar imaging”, Digital Signal Processing, vol. 144, p. 104274, 2024, doi: 10.1016/j.dsp.2023.104274.

**[2]**  J. Wang, P. Aubry and A. Yarovoy, "3-D Short-Range Imaging With Irregular MIMO Arrays Using NUFFT-Based Range Migration Algorithm," in  _IEEE Transactions on Geoscience and Remote Sensing_, vol. 58, no. 7, pp. 4730-4742, July 2020, doi: 10.1109/TGRS.2020.2966368.

**[3]**  Jianping Wang, January 10, 2020, "EM data acquired with irregular planar MIMO arrays", IEEE Dataport, doi:  [https://dx.doi.org/10.21227/src2-0y50](https://dx.doi.org/10.21227/src2-0y50).

## FAQs: Codes
### Main Files:
- **For simulations**, follow the instructions provided in [main-simulated.ipynb](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/main-simulated.ipynb).
- **For the experimental reconstructions**, follow the instructions provided in [main-experimental.ipynb](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/main-experimental.ipynb).

### Are you looking for the, 
**- Codes for the reconstruction algorithms:** You can find them in [../src/optimization.py](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/src/optimization.py) (apart from the comments present in [optimization.py](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/src/optimization.py), further explanations are provided in [main-simulated.ipynb](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/main-simulated.ipynb) and [main-experimental.ipynb](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/main-experimental.ipynb)).
 
**- Deep denoiser:** Check [../src/nn_models/unet3D.py](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/src/nn_models/unet3D.py) for the network class. The parameters of the utilized model architecture can be found in [../trained_model/info.json](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/trained_model/info.json). For the parameter dictionary of the trained model, check [../trained_model/base_model.pt`](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/trained_model/base_model.pt).

**- Observation model:** Check [../src/forward_models/bornapprox.py](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/src/forward_models/bornapprox.py) for the forward model class. The base model is  `MVprod()`, but if you have memory problems, you may prefer to use `MVProd_iterK()`, which computes the Born approximated measurements iteratively. 

### Some remarks on,
**- Parallel processing:** All solvers can work on CPUs and GPUs. You can process the measurements in parallel and also asynchronously (see [main-simulated.ipynb](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/main-simulated.ipynb) for further info).  If you plan on parallel processing batch of measurements, then I suggest you to enable 'asynchronous' argument of the solvers as this allows to asynchronously stop the image updates and speeds up the execution. On the other hand, if set to False, the solver will continue the optimization even after some images in the batch converge, which can potentially create numerical instability.

**- Scale invariance:** For the deep neural network, you should use the `scale_invariance_wrapper()` class if the measurements have a different scale than the training data.

**- Using your DNN:** The NNCG-CSALSA solver allows you to use both blind and non-blind denoisers. Please check the notebooks for further information.
## CITATION
Please cite the following when using this code:

    @ARTICLE{oral2024plug,
    author={Oral, Okyanus and Oktem, Figen S.},
    journal={IEEE Transactions on Computational Imaging}, 
    title={Plug-and-Play Regularization on Magnitude With Deep Priors for 3D Near-Field MIMO Imaging}, 
    year={2024},
    volume={10},
    number={},
    pages={762-773},
    doi={10.1109/TCI.2024.3396388}}
   
## Other Results

![img](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/images/SimulatedTop.png "Simulated Reconstructions")
![img](https://github.com/METU-SPACE-Lab/PnP-Regularization-on-Magnitude/blob/main/images/SimulatedResults.png "Simulated Reconstructions")

## CONTACT:
If you have any questions or need help, please feel free to contact me ([Okyanus Oral](https://ookyanus.github.io), **email:**  ookyanus@metu.edu.tr).
