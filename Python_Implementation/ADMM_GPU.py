'''
ADMM GPU implementation - The parameters for 
each scale (N) and model (HSC or BDS) are mentioned
in the supplement of the manuscript. Please change the
parameters according to the scale (N). 
'''

!nvcc --version
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
!mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
!wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
!dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
!apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
!apt-get update
!apt-get -y install cuda
!nvcc --version

import matplotlib.pyplot as plt
!pip install pyfftw
import scipy.io as sio
import seaborn
import scipy
import pyfftw
import time
import pandas as pd
import numpy as np

!pip uninstall cupy
!pip install cupy --no-cache-dir

import cupy as cp
from cupy.linalg import norm

scale = 64

class FFTW2d_picks_gpu:
    def __init__( self,picks, n, axes = (0,1)):
        self.axes = axes
        self.picks = picks
        self.n = n
        self.n_total = n*n
        self.new_img = cp.zeros([self.n,self.n],dtype=cp.complex128)

    # let's call k-space <- image as forward
    def forward( self, im):
        return cp.fft.ifft2(im)[self.picks, self.picks]

    # let's call image <- k-space as backward
    def backward( self, ksp ):
        self.new_img[np.ix_(self.picks, self.picks)] = ksp
        return cp.fft.fft2(self.new_img)/self.n_total

"""
fast ADMM recovery
"""
N = scale

def get_I_matrix(N):
    return cp.diag(cp.ones(N))
  
def get_P_matrix(picks, N):
    return get_I_matrix(N)[cp.ix_(picks),:]
  
def get_M_hat_diag(N, beta, picks):
    M_hat_diag = beta*cp.ones(N*N)
    P_diag = cp.zeros(N)
    P_diag[picks] += 1
    M_hat_diag = M_hat_diag + cp.kron(P_diag, P_diag)
    return M_hat_diag
  
def get_a_hat_vec(a_hat, z, beta):
    return cp.ravel((a_hat + cp.fft.ifft2(beta*z)),order='F') # column first
  
def proximal_l1_soft_thresh( x0, th ):
    a_th = cp.abs(x0) - th
    a_th[a_th<0] = 0
    return cp.multiply(cp.exp(1j*cp.angle(x0)), a_th)
  
def ADMM_gpu2(Afunc, 
              invAfunc, 
              B, 
              picks, 
              _lambda,
              beta, 
              Nite,
              printing = False,
              save_err = False,
              true_sig = None,
              abs_error = 1e-2,
              rel_error = 1e-3):
    
    z = cp.zeros(x.shape)
    N, _ = z.shape
    
    n_total = N*N
    
    a_hat = cp.zeros([N, N])
    a_hat = a_hat.astype(complex)
    a_hat[cp.ix_(picks, picks)] = B
    
    M_hat_diag = get_M_hat_diag(N, beta, picks)
    a_hat_vec = get_a_hat_vec(a_hat, z, beta)
      
    # u_old and u_new
    u_old = cp.fft.fft2((a_hat_vec / M_hat_diag).reshape((N, N), order = 'F'))
    u_new = cp.copy(u_old)

    
    z_new = z
    z_old = cp.copy(z_new)
    
    y = cp.ones(u_old.shape)
    y = y.astype(float)
    
    
    for idx in range(Nite):
        z_new = proximal_l1_soft_thresh(u_old + y/beta, _lambda/beta)

        # part II
        a_hat_vec = get_a_hat_vec(a_hat, z_new, beta)

        u_new = cp.fft.fft2((a_hat_vec / M_hat_diag).reshape((N, N), order = 'F'))


        #part III: update lambda
        y = y - beta*(z_new-u_new)

        
        #stop condition; check on every 50 iterations
        if idx % 100 == 1: 
            r_k = norm(u_new - z_new)
            s_k = norm(-beta*(z_new - z_old))
            
            err_pri = (N**2.0)*abs_error + rel_error*max(norm(u_new), norm(z_new)) 
            err_dual = (N**5.0)*abs_error + rel_error*norm(_lambda)

            if r_k < err_pri and s_k < err_dual:
                print("meet stop critera")
                if printing: # debug_mode
                    if true_sig is None: print("error! no true signal as icput")
                    else: print(norm(u_new.real/n_total - true_sig.real), idx)  
                return u_old/n_total
        u_old = cp.copy(u_new)
        z_old = cp.copy(z_new)
    print("reached max iteration")
    return u_old/n_total




rel_errors = []
max_abs_errors = []
max_rel_errors = []
timing_gpu = []
beta=0.005
nums_sampled = 5

'''
Loading the data that has been generated in R for a given scale.
'''
file_1 = str(scale)+"_"+ str(1)+".xlsx"

#Loading the full signal which is same for all datasets within the same scale.
signal_real = pd.read_excel(file_1, sheet_name='signal_real')
signal_im = pd.read_excel(file_1, sheet_name='signal_im')
signal_real = cp.asarray(signal_real)
signal_im = cp.asarray(signal_im)
x = signal_real + 1j*signal_im
N,_ = x.shape

#fft_x = scipy.fft.fft2(x)/(len(x)**2)
fft_x = pd.read_excel(file_1, sheet_name='trans_prob')
fft_x = cp.asarray(fft_x)

for i in range(1,nums_sampled+1):
  file = str(scale)+"_"+ str(i)+".xlsx"

  #Loading the partially sampled FFT matrix
  phi2d_real = pd.read_excel(file, sheet_name='phi2d_real')
  phi2d_im = pd.read_excel(file, sheet_name='phi2d_im')
  phi2d_real = cp.asarray(phi2d_real)
  phi2d_im = cp.asarray(phi2d_im)
  A = phi2d_real + 1j*phi2d_im

  #Loading the indices and substracting by 1 because unlike R, Python
  #indexing starts from 0. 
  
  indices = pd.read_excel(file, sheet_name='indices_subsampled')
  indices = indices['indices'].to_list()
  picks = [number - 1 for number in indices]
  M = len(indices)
  _lambda = 0.5*(np.log(M))

  #Loading the partial observation signal matrix
  signal_subsampled_real = pd.read_excel(file, sheet_name='signal_subsampled_real')
  signal_subsampled_im = pd.read_excel(file, sheet_name='signal_subsampled_im')
  signal_subsampled_real = cp.asarray(signal_subsampled_real)
  signal_subsampled_im = cp.asarray(signal_subsampled_im)
  B = signal_subsampled_real + 1j*signal_subsampled_im
  #n,_ = A.shape

  Aopt = FFTW2d_picks_gpu(picks, N)
  start_time = time.time()
  u = ADMM_gpu2(Aopt.forward, 
                Aopt.backward, 
                B, 
                picks = picks, 
                _lambda = _lambda,
                Nite = 600, 
                beta = beta)
  rel_errors.append((norm(u.real-fft_x, 2))/(norm(fft_x, 2)))
  diff = (fft_x - u.real)
  max_abs_errors.append(diff.max().max())
  max_rel_errors.append(diff.max().max()/fft_x.max().max())  


  #errors_gpu.append((norm(u.real-fft_x, 2))/(norm(fft_x, 2)))
  timing_gpu.append(time.time() - start_time)

  final_df_gpu = pd.DataFrame(list(zip([beta]*nums_sampled, timing_gpu,max_abs_errors, rel_errors, max_rel_errors)),
               columns =["stepsize (beta)","timing gpu", "max abs errors ADMM gpu", "relative errors ADMM gpu", "max relative errors ADMM gpu"])

temp_file = str(scale)+"_timing_errors_ADMM_gpu"+".csv"

final_df_gpu.to_csv(temp_file, index = False, header=True)