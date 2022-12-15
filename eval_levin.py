import torch
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from PIL import Image
from utils.iterative_scheme import iterative_scheme
from utils.utils_test import shift_inv_metrics
from utils.utils_torch import p4ip_wrapper
from models.network_p4ip import P4IP_Net
from models.network_p4ip_denoiser import P4IP_Denoiser


ALPHA = 20.0 # Photon level for noise


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(20)


blind_psnr = np.zeros([2,4,8])
non_blind_psnr = np.zeros([2,4,8])

# # Load Non-Blind Solver
MODEL_FILE = 'model_zoo/p4ip_100epoch.pth'
p4ip = P4IP_Net(n_iters = 8); p4ip.load_state_dict(torch.load(MODEL_FILE))
p4ip.to(device); p4ip.eval()
# # Load Poisson Denoiser
MODEL_FILE = 'model_zoo/denoiser_p4ip_100epoch.pth'
denoiser = P4IP_Denoiser(n_iters = 8); denoiser.load_state_dict(torch.load(MODEL_FILE))
denoiser.to(device); denoiser.eval()


for IM_IDX in range(1,5):
	for K_IDX in range(1,9):

		yn = np.asarray(Image.open('data/im'+str(IM_IDX)+'_kernel'+str(K_IDX)+'_img.png'), dtype=np.float32)
		yn = yn/255.0
		x = np.asarray(Image.open('data/im'+str(IM_IDX)+'.png'), dtype=np.float32)
		x = x/255.0
		kernel = np.asarray(Image.open('data/kernel'+str(K_IDX)+'.png'), dtype=np.float32)
		kernel = kernel/np.sum(kernel.ravel())
		if yn.ndim > 2:
			yn = np.mean(yn,axis=2)
			x = np.mean(x,axis=2)
		# Noisy + Blurred Image here
		y = np.random.poisson(np.maximum(ALPHA*yn,0)).astype(np.float32)

		# Non-Blind Deconvolution using the ground-truth Kernel
		x_non_blind = p4ip_wrapper(y, kernel, ALPHA, p4ip, 'symmetric')
		non_blind_psnr[:, IM_IDX-1, K_IDX-1] = shift_inv_metrics(x_non_blind, x)

		# Blind-Deconvolution which estimates the kernel
		x_list, k_list = iterative_scheme(y, ALPHA, p4ip, denoiser, {'VERBOSE': True})
		x_blind = x_list[-1]
		blind_psnr[:, IM_IDX-1, K_IDX-1] = shift_inv_metrics(x_blind, x)



non_blind_avg = np.mean(non_blind_psnr, axis=(1,2))
blind_avg = np.mean(blind_psnr, axis=(1,2))

print('Non-Blind Deconvolution, Average PSNR / SSIM: %0.2f / %0.3f'%(non_blind_avg[0],non_blind_avg[1]))
print('Blind Deconvolution, Average PSNR / SSIM: %0.2f / %0.3f'%(blind_avg[0], blind_avg[1]))