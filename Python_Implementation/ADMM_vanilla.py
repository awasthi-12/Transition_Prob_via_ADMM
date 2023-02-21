'''
ADMM Vanilla implementation - The parameters for 
each scale (N) and model (HSC or BDS) are mentioned
in the supplement of the manuscript. Please change the
parameters according to the scale (N). 
'''

!pip install "openpyxl==3.0.0"
!pip install pyfftw

import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn
import numpy as np
import scipy
import pyfftw
import sys
import multiprocessing
import pandas as pd
import time
from numpy.linalg import norm
pyfftw.interfaces.cache.enable()


#2-D FFT Implementation using pyFFTW

def fftw2d( data, axes = (0,1), threads = 1):
    a = pyfftw.empty_aligned(data.shape, dtype=data.dtype)
    b = pyfftw.empty_aligned(data.shape, dtype=data.dtype)

    fft_object = pyfftw.FFTW(a, b, axes=axes, threads=threads)
    a[:] = data
    iftdata = fft_object()
    return iftdata

def ifftw2d( data, axes = (0,1), threads = 1 ):
    b = pyfftw.empty_aligned(data.shape, dtype=data.dtype)
    c = pyfftw.empty_aligned(data.shape, dtype=data.dtype)

    ifft_object = pyfftw.FFTW(b, c, axes=axes, direction='FFTW_BACKWARD', threads=threads)
    b[:] = data
    return ifft_object()

class FFTW2d_picks:
    "this is ndim FFTW for MRI recon"
    def __init__(self,picks, n, axes = (0,1), threads = 1):
        self.axes = axes
        self.threads = threads
        self.picks = picks
        self.n = n
        self.n_total = n*n
        self.new_img = np.zeros([self.n,self.n],dtype=complex)

    #let's call k-space <- image as forward
    def forward(self, im):
        return ifftw2d(im)[self.picks, self.picks]

    # let's call image <- k-space as backward
    def backward(self, ksp):
        self.new_img[np.ix_(self.picks, self.picks)] = ksp
        return fftw2d(self.new_img)/self.n_total

 #ADMM with primal and dual stopping criteria

import pyfftw
from numpy.linalg import norm
import pdb

"""
fast ADMM recovery
"""

def get_I_matrix(N):
    return np.diag(np.ones(N))
  
def get_P_matrix(picks, N):
    return get_I_matrix(N)[np.ix_(picks),:]
  
def get_M_hat_diag(N, beta, picks):
    M_hat_diag = beta*np.ones(N*N)
    P_diag = np.zeros(N)
    P_diag[picks] += 1
    M_hat_diag = M_hat_diag + np.kron(P_diag, P_diag)
    return M_hat_diag
  
def get_a_hat_vec(a_hat, z, beta):
    return (a_hat + ifftw2d(beta*z)).flatten('F') # column first
  
def proximal_l1_soft_thresh( x0, th ):
    a_th = np.abs(x0) - th
    a_th[a_th<0] = 0
    return np.multiply(np.exp(1j*np.angle(x0)), a_th)
  
def ADMM_fast(Afunc, 
              invAfunc, 
              B, 
              picks, 
              beta, 
              Nite, 
              _lambda,
              printing = False,
              save_err = False,
              true_sig = None,
              abs_error = 1e-2,
              rel_error = 1e-3):
    
    z = np.zeros(x.shape,dtype = complex)
    N, _ = z.shape
    
    n_total = N*N
    
    a_hat = np.zeros([N, N])
    a_hat = a_hat.astype(complex)
    a_hat[np.ix_(picks, picks)] = B
    
    m_hat_prac = get_M_hat_diag(N, beta, picks) # correct
    a_hat_vec = get_a_hat_vec(a_hat, z, beta)
    
    # u_old and u_new
    u_old = fftw2d((a_hat_vec / m_hat_prac).reshape((N, N), order = 'F'))
    u_new = np.copy(u_old)
    
    z_new = np.zeros(x.shape,dtype = complex)
    z_old = np.copy(z_new)
    
    y = np.ones(u_old.shape)
    y = y.astype(float)
    

    for idx in range(Nite):
      z_new = proximal_l1_soft_thresh(u_old + y/beta, _lambda/beta)

      # part II
      a_hat_vec = get_a_hat_vec(a_hat, z_new, beta)
      u_new = fftw2d((a_hat_vec / m_hat_prac).reshape((N, N), order = 'F'), threads = 32)
            
      # part III: update y
      y = y - beta*(z_new-u_new)

      # stop condition; check on every 100 iterations
      if idx % 100 == 1: 
        r_k = norm(u_new - z_new)
        s_k = norm(-beta*(z_new - z_old))
        
        err_pri = (N**2.0)*abs_error + rel_error*max(norm(u_new), norm(z_new))
        err_dual = (N**5.0)*abs_error + rel_error*norm(y)
        if r_k < err_pri and s_k < err_dual:
          print("meet stop critera")
          if printing: #debug_mode
             if true_sig is None: print("error! no true signal as input")
             else: print(norm(u_new.real/n_total - true_sig.real,2), idx)  
          return u_old/n_total
      u_old = np.copy(u_new)
      z_old = np.copy(z_new)
    print("reached max iteration")
    return u_old/n_total

scale = 64 #Change the scale for according to the maximum population size.
nums_sampled = 5
rel_errors = []
max_abs_errors = []
max_rel_errors = []
timing = []
beta=0.08

'''
Loading the data that has been generated in R using
Jason's code for a given scale (N value). This data 
is available in the Github repository under Data.
'''

file_1 = str(scale)+"_"+ str(1)+".xlsx"

#Loading the full signal which is same for all datasets within the same scale.
signal_real = pd.read_excel(file_1, sheet_name='signal_real')
signal_im = pd.read_excel(file_1, sheet_name='signal_im')
signal_real = np.asarray(signal_real)
signal_im = np.asarray(signal_im)
x = signal_real + 1j*signal_im
N,_ = x.shape

#fft_x = scipy.fft.fft2(x)/(len(x)**2)
fft_x = pd.read_excel(file_1, sheet_name='trans_prob')

for i in range(1,nums_sampled+1):
  file = str(scale)+"_"+ str(i)+".xlsx"

  #Loading the partially sampled FFT matrix
  phi2d_real = pd.read_excel(file, sheet_name='phi2d_real')
  phi2d_im = pd.read_excel(file, sheet_name='phi2d_im')
  phi2d_real = np.asarray(phi2d_real)
  phi2d_im = np.asarray(phi2d_im)
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
  signal_subsampled_real = np.asarray(signal_subsampled_real)
  signal_subsampled_im = np.asarray(signal_subsampled_im)
  B = signal_subsampled_real + 1j*signal_subsampled_im


  Aopt = FFTW2d_picks(picks, N, threads = 32)
  start_time = time.time()
  u = ADMM_fast(Aopt.forward,
                Aopt.backward,
                B,
                picks = picks, 
                _lambda = _lambda,
                Nite = 600,
                beta = beta)
  timing.append(time.time() - start_time)
  rel_errors.append((np.linalg.norm(u.real-fft_x))/(np.linalg.norm(fft_x)))
  diff = (fft_x - u.real)
  max_abs_errors.append(diff.max().max())
  max_rel_errors.append(diff.max().max()/fft_x.max().max())
  
final_df = pd.DataFrame(list(zip([beta]*nums_sampled,timing,max_abs_errors, rel_errors, max_rel_errors)),
               columns =["stepsize (beta)", "timing", "max abs errors ADMM", "relative errors ADMM", "max relative errors ADMM"])

temp_file = str(scale)+"_timing_errors_ADMM"+".csv"

final_df.to_csv(temp_file, index = False, header=True)