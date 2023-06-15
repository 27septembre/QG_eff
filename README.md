# Code and data for the paper

## 1. Generate data 
* calculate the PV flux, its divergence, and eddy force function
* need FEniCS (version 2019.1.0) download [here]("https://fenicsproject.org/download/")
* `qgm2_parameters.py` provides parameters

## 2. CNN Training
* can directly run at the Colab or locally
* test data has been given in `data/training/00/`

## 3. Plot script
* all the plots in the paper, except for Fig.x
* code to create csv file for ensemble members 

## 4. for Fig.x
* test data has been given in `data/preds/`
  
## 5. Noise data
* generate noise data into `data/noise_filtered/`
* script 2 for training
  
## data directory structure:
[data]  
- [preds] `(predicted by CNN in script 2.)`  
  - [00]  `(for default setting in qg model)` 
    - [0] `(layer 0 : surface)` 
      - Pr_uvq_empb_00_psi_30.npz
      - Pr_div_empb_00_psi_30.npy
      - Pr_grad_empb_00_psi_30.npz
- [noise_filtered] `(for script 5. and 2.)`  
- [for_plot] `(for script 3.)`  
  - l2_00_paper.csv
  - l2_00_noise_psi.csv
- [outputs] `(data from qg model in script 1. )`  
  - u_v_int.dat
  - v_v_int.dat
  - r_int.dat
  - p_l_int.dat
  - s_int.dat
  - u_u_int.dat
  - v_int.dat
  - psi_int.dat
  - u_int.dat
  - q_int.dat
- [training] `(outputs from qg model in script 1. , test data for script 2.)`  
  - [00]
    - div_uq_empb_80.npy
    - grad_empb_80.npz
    - uvq_empb_80.npz 
