"""
Neuroimaging cartesian reconstruction
=====================================

Credit: S Lannuzel, L Elgueddari

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 3D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct_3D.fourier import NFFT3, NUFFT, FFT3
from pysap.plugins.mri.reconstruct_3D.utils import imshow3D
from pysap.plugins.mri.parallel_mri.gradient import Gradient_pMRI
from pysap.plugins.mri.reconstruct_3D.linear import pyWavelet3
from pysap.plugins.mri.reconstruct.linear import Wavelet2
from pysap.plugins.mri.reconstruct_3D.utils import normalize_samples
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu
from pysap.plugins.mri.reconstruct_3D.utils import convert_mask_to_locations_3D
from pysap.plugins.mri.reconstruct_3D.utils import convert_locations_to_mask_3D
from modopt.math.metrics import ssim, psnr

# Third party import
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.fftpack as pfft


def mat2grey(x):
    return (np.abs(x) - np.min(np.abs(x)))/(np.max(np.abs(x))
                                            - np.min(np.abs(x)))

# Load input data
# Il = get_sample_data("3d-pmri")
# Iref = np.squeeze(np.sqrt(np.sum(np.abs(Il)**2, axis=0)))
# Iref.astype(np.complex64)

Iref = loadmat('/volatile/bsarthou/datas/meas_MID14_gre_800um_iso_128x128x128_FID24.mat')['ref']
Iref.astype(np.complex64)
# Iref = np.swapaxes(Iref, 1, 0)
imshow3D(Iref, display=True)

# samples = get_sample_data("mri-radial-3d-samples").data
samples = loadmat('/volatile/bsarthou/datas/samples_sparkling_3D_N128_502x1536x8_FID4971.mat')['samples']
samples = normalize_samples(samples)

# Full Cartesian sampling
# samples = convert_mask_to_locations_3D(np.ones(Iref.shape))


#############################################################################
# Generate the kspace
# -------------------
#
# From the 3D phantom and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace
fourier_op_gen = NFFT3(samples=samples, shape=Iref.shape)
# fourier_op_gen = NUFFT(samples=samples, shape=Iref.shape, platform='cpu')
# fourier_op_gen = FFT3(samples=samples, shape=Iref.shape)

kspace_data = fourier_op_gen.op(Iref)

# Zero order solution
image_rec0 = fourier_op_gen.adj_op(kspace_data)
imshow3D(np.abs(image_rec0), display=True)

max_iter = 500

linear_op = pyWavelet3(wavelet_name="sym8",
                       nb_scale=3)

# linear_op = Wavelet2(
#         nb_scale=2,
#         wavelet_name='ATrou3D')

fourier_op = NFFT3(samples=samples, shape=Iref.shape)
# fourier_op = NUFFT(samples=samples, shape=Iref.shape, platform='gpu')
# fourier_op = FFT3(samples=samples, shape=Iref.shape)

print('Starting Lipschitz constant computation')

gradient_op = Gradient_pMRI(data=kspace_data,
                            fourier_op=fourier_op,
                            linear_op=linear_op)

print('Lipschitz constant found: ', str(gradient_op.spec_rad))
mu = 5e-1
x_final, transform = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    mu=mu,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1,
    get_cost=False)

# print(np.abs(x_final))
imshow3D(np.abs(x_final), display=True)
# plt.figure()
# plt.plot(cost)
# plt.show()

np.save('/volatile/bsarthou/datas/save_baboon_128_NUFFT_cubic_GPU_sp3d.npy', x_final)
print('FISTA Mu:{} SSIM: {}'.format(mu, ssim(mat2grey(Iref), mat2grey(x_final),
                                    None)))
print('FISTA Mu:{} PSNR: {}'.format(mu, psnr(mat2grey(Iref), mat2grey(x_final),
                                    None)))

#
# gradient_op_cd = Gradient_pMRI(data=kspace_data,
#                                fourier_op=fourier_op)
# x_final, transform, cost = sparse_rec_condatvu(
#     gradient_op=gradient_op_cd,
#     linear_op=linear_op,
#     std_est=None,
#     std_est_method=None,
#     std_thr=2.,
#     mu=mu,
#     tau=None,
#     sigma=None,
#     relaxation_factor=1.0,
#     nb_of_reweights=0,
#     max_nb_of_iter=max_iter,
#     add_positivity=False,
#     atol=1e-4,
#     verbose=1,
#     get_cost=True)
# imshow3D(np.abs(x_final), display=True)
# plt.figure()
# plt.plot(cost)
# plt.show()
#
# print('CONDAT Mu:{} SSIM: {}'.format(mu, ssim(mat2grey(Iref),
#                                               mat2grey(x_final),
#                                      None)))
