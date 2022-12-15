import numpy as np
from skimage.metrics import structural_similarity as ssim


def shift_inv_metrics(x1, x2, search_window=20):
	max_psnr = -np.inf
	max_ssim = -np.inf
	for i in np.arange(-search_window, search_window+1):
		for j in np.arange(-search_window, search_window+1):
			x0 = np.roll(x1, (i,j), axis=[0,1])
			mse = np.mean((x0-x2)**2)
			psnr =  -10*np.log10(mse)
			ssim_val = ssim(x2, x0, data_range=np.max(np.ravel(x0))-np.min(np.ravel(x0)))

			if psnr > max_psnr:
				max_psnr = psnr
			if ssim_val > max_ssim:
				max_ssim = ssim_val

	return [max_psnr, max_ssim]
