from pysap.plugins.mri.reconstruct.fourier import NFFT2
from pysap.plugins.mri.reconstruct.reconstruct import sparse_rec_fista
from pysap.plugins.mri.reconstruct.reconstruct import sparse_rec_condatvu
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
from pysap.plugins.mri.parallel_mri.gradient import Gradient_pMRI
# from pysap.plugins.mri.reconstruct_3D.gradient import GradSynthesis3
# from pysap.plugins.mri.reconstruct_3D.gradient import GradAnalysis3
from pysap.plugins.mri.reconstruct_3D.linear import pyWavelet3
from pysap.plugins.mri.reconstruct.linear import Wavelet2
from pysap.plugins.mri.reconstruct_3D.fourier import NFFT3
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista as fista_3D
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu as condat_3D


# Third party import
import numpy as np
import scipy.fftpack as pfft


def recons_pysap(image, samples_loc, **kwargs):
    # Define observation in k-space
    fourier_op = NFFT2(samples=samples_loc, shape=image.shape)
    kspace_obs = fourier_op.op(image)

    x_final, transform = sparse_rec_condatvu(
        data=kspace_obs,
        samples=samples_loc,
        **kwargs)

    cost = 0

    return x_final.data, cost


def recons_pysap_3D(image, samples_loc, **kwargs):

    linear_op = pyWavelet3(wavelet_name="bior6.8",
                           nb_scale=4)
    fourier_op = NFFT3(samples=samples_loc, shape=image.shape)
    kspace_obs = fourier_op.op(image)
    gradient_op_cd = Gradient_pMRI(data=kspace_obs,
                                   fourier_op=fourier_op)

    x_final, transform, cost = condat_3D(
                                        gradient_op=gradient_op_cd,
                                        linear_op=linear_op,
                                        **kwargs)

    return x_final.data, cost
