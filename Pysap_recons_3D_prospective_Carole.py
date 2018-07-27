"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L Elgueddari, S.Lannuzel

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

"""

# Package import
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct_3D.fourier import NFFT3, NUFFT
from pysap.plugins.mri.reconstruct_3D.utils import imshow3D
from pysap.plugins.mri.reconstruct_3D.linear import pyWavelet3
from pysap.plugins.mri.parallel_mri.gradient import Gradient_pMRI
from pysap.plugins.mri.reconstruct_3D.utils import normalize_samples
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu

# Third party import
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from modopt.math.metrics import ssim
# Loading input data
# Il = get_sample_data("3d-pmri")
# Iref = np.squeeze(np.sqrt(np.sum(np.abs(Il)**2, axis=0)))
# Smaps = np.asarray([Il[channel]/Iref for channel in range(Il.shape[0])])


def mat2gray(x):
    return (np.abs(x)-np.abs(x).min())/(np.abs(x).max()-np.abs(x).min())

Smaps = loadmat('/volatile/bsarthou/datas/XP_pysap/data_N320_sos_vds_nc1139/Smaps.mat')['Smaps']
Smaps = np.moveaxis(Smaps, -1, 0)
# imshow3D(Iref, display=True)

# samples = get_sample_data("mri-radial-3d-samples").data
samples = loadmat('/volatile/bsarthou/datas/XP_pysap/data_N320_sos_vds_nc1139/samples.mat')['samples']
samples = normalize_samples(samples)

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace

# gen_fourier_op = NFFT3(samples=samples,
#                        shape=(128, 128, 160))

print('Generate the k-space')

# kspace_data = np.asarray([gen_fourier_op.op(Il[channel]) for channel
#                           in range(Il.shape[0])])

kspace_data = loadmat('/volatile/bsarthou/datas/XP_pysap/data_N320_sos_vds_nc1139/datavalues.mat')['datavalues']
kspace_data = np.moveaxis(kspace_data, -1, 0)
print(kspace_data.shape)
#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the FISTA reconstruction
max_iter = 200
linear_op = pyWavelet3(wavelet_name="sym8",
                       nb_scale=3)

# fourier_op = NFFT3(samples=samples,
#                    shape=(128, 128, 128))
fourier_op = NUFFT(samples=samples, shape=(152, 152, 104), platform='gpu')

print('Generate the zero order solution')

rec_0 = np.asarray([fourier_op.adj_op(kspace_data[l, :]) for l in range(32)])
imshow3D(np.squeeze(np.sqrt(np.sum(np.abs(rec_0)**2, axis=0))),
         display=True)

gradient_op = Gradient_pMRI(data=kspace_data,
                            fourier_op=fourier_op,
                            linear_op=linear_op,
                            S=Smaps)

x_final, transform, cost = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    mu=0,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1,
    get_cost=True)
imshow3D(np.abs(x_final), display=True)


plt.figure()
plt.plot(cost)
plt.show()

print('Saving the cube')
np.save('/volatile/bsarthou/datas/XP_pysap/save_FISTA_300_sym8_mu5e-6.npy', x_final)

ref = np.load('/volatile/bsarthou/datas/XP_pysap/ref_Ipat4.npy')

print('SSIM iPAT4:', ssim(mat2gray(ref), mat2gray(x_final), mask=None))

#
# #############################################################################
# # Condata-Vu optimization
# # -----------------------
# #
# # We now want to refine the zero order solution using a Condata-Vu
# # optimization.
# # Here no cost function is set, and the optimization will reach the
# # maximum number of iterations. Fill free to play with this parameter.
#
# # Start the CONDAT-VU reconstruction
# max_iter = 1
# gradient_op_cd = Gradient_pMRI(data=kspace_data,
#                                fourier_op=fourier_op,
#                                S=Smaps)
# x_final, transform = sparse_rec_condatvu(
#     gradient_op=gradient_op_cd,
#     linear_op=linear_op,
#     std_est=None,
#     std_est_method="dual",
#     std_thr=2.,
#     mu=0,
#     tau=None,
#     sigma=None,
#     relaxation_factor=1.0,
#     nb_of_reweights=0,
#     max_nb_of_iter=max_iter,
#     add_positivity=False,
#     atol=1e-4,
#     verbose=1)
#
# imshow3D(np.abs(x_final), display=True)
