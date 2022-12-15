import numpy as np
from numpy.linalg import norm
from numpy.fft import fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from scipy.ndimage import sobel
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_torch import conv_fft, conv_fft_batch, psf_to_otf, img_to_tens, scalar_to_tens, p4ip_wrapper, unet_wrapper, p4ip_denoiser, sharpness
from utils.utils_deblur import shock, mask_gradients, psf2otf, otf2psf, imresize, D, Dt, shrinkage
from models.network_p4ip import P4IP_Net
from models.ResUNet import ResUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def shrinkage_torch(x, rho):
	return F.relu(x-rho) - F.relu(-x-rho)

class Normalize_Kernel(nn.Module):
	def __init__(self):
		super(Normalize_Kernel, self).__init__()
		self.relu = nn.ReLU()
	def forward(self, k):
		k = self.relu(k) 
		k_sum = torch.sum(k)
		k = k/k_sum
		return k


	
