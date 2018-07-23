"""
Benchmarking (3D) wavelets from PySAP
=====================================

Credit: B. Sarthou

This script allows you to test on several iterations the
decomposition/recomposition time and the error of each wavelet, and to compare
the performance two by two.
As entry, you give a list of dictionary with the name and the object of the
operator. You can also define the number of iterations for testing,
the number of scale, the input image, and the output savefile for
each couple of wavelets

This script can work in 2D if your wavelets are correctly defined, and the
image is 2D
"""

# Package import
from pysap.plugins.mri.reconstruct_3D.utils import imshow3D
from pysap.plugins.mri.reconstruct_3D.utils import normalize_samples
from pysap.plugins.mri.reconstruct_3D.utils import convert_mask_to_locations_3D
from pysap.plugins.mri.parallel_mri.gradient import Gradient_pMRI

from pysap.plugins.mri.reconstruct_3D.linear import pyWavelet3
from pysap.plugins.mri.reconstruct.linear import Wavelet2
from pysap.plugins.mri.reconstruct_3D.fourier import NFFT3
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu


# Third party import
import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time

nb_scale = 3
R = 10

# If you want to use wavelets from PyWavelet3 or Sparse3D binding, you can
# use those lists. if not, you can directly define your wavelet operator in
# the list obj_wt by putting a dictionary for each wavelet, with a name,
# and the operator
list_pwt = ['bior6.8', 'db4', 'bior2.4', 'bior3.3']
list_sp3d = [{'name': 'BiOr79', 'sp3d_name': 'BiOrthogonalTransform3D',
              'filter_id': 1},
             {'name': 'Db4', 'sp3d_name': 'BiOrthogonalTransform3D',
              'filter_id': 2},
             {'name': 'Bior35', 'sp3d_name': 'BiOrthogonalTransform3D',
              'filter_id': 12},
             {'name': 'Bior44', 'sp3d_name': 'BiOrthogonalTransform3D',
              'filter_id': 13}]

obj_wt_pwt = [{'name': 'pwt_'+ite,
               'wt': pyWavelet3(wavelet_name=ite,
                                nb_scale=nb_scale)} for ite in list_pwt]

obj_wt_sp3d = [{'name': 'sp3d_'+ite['name'],
                'wt': Wavelet2(wavelet_name=ite['sp3d_name'],
                               nb_scale=nb_scale-1,
                               **{'type_of_filters': ite['filter_id']})}
               for ite in list_sp3d]

# This is a list of wavelets, each item is a dic containing a key 'name' and a
# key 'wt' containing the object of the wt
obj_wt = obj_wt_pwt + obj_wt_sp3d

# Load input data (format: 3D complex numpy array)
filename = '/volatile/bsarthou/datas/' \
            'meas_MID14_gre_800um_iso_128x128x128_FID24.mat'
Iref = loadmat(filename)['ref']

for (i, dic_wt_i) in enumerate(obj_wt):
    for j in range(i+1, len(obj_wt)):
        dic_wt_j = obj_wt[j]
        wt_i = dic_wt_i['wt']
        wt_j = dic_wt_j['wt']
        print('Testing {} against {}'.format(dic_wt_i['name'],
                                             dic_wt_j['name']))

        time_i_trans, time_i_recons, error_i = [], [], []
        for k in range(R):
            linear_op = wt_i
            start = time.clock()
            coeffs = linear_op.op(Iref)
            mid = time.clock()
            recons1 = linear_op.adj_op(coeffs)
            end = time.clock()
            time_i_trans.append(mid - start)
            time_i_recons.append(end - mid)
            error_i.append(np.abs((Iref-recons1).mean()))
            # print('Trans time:', mid - start, 'Recons time:', end - mid)

        print('{}: Trans, recons, error=({},{}, {})'.format(
                                                      dic_wt_i['name'],
                                                      np.mean(time_i_trans),
                                                      np.mean(time_i_recons),
                                                      np.mean(error_i)))

        time_j_trans, time_j_recons, error_j = [], [], []

        for k in range(R):
            linear_op = wt_j
            start = time.clock()
            coeffs = linear_op.op(Iref)
            mid = time.clock()
            recons2 = linear_op.adj_op(coeffs)
            end = time.clock()
            time_j_trans.append(mid - start)
            time_j_recons.append(end - mid)
            error_j.append(np.abs((Iref-recons2).mean()))

            # print('Trans time:', mid - start, 'Recons time:', end - mid)
        print('{}: Trans, recons, error=({},{}, {})'.format(
                                                     dic_wt_j['name'],
                                                     np.mean(time_j_trans),
                                                     np.mean(time_j_recons),
                                                     np.mean(error_j)))
        results = {'name_i': dic_wt_i['name'], 'name_j': dic_wt_j['name'],
                   't_i': [np.mean(time_i_trans),
                           np.mean(time_i_recons)],
                   't_j': [np.mean(time_j_trans),
                           np.mean(time_j_recons)],
                   't_diff': [np.mean(time_i_trans) -
                              np.mean(time_j_trans),
                              np.mean(time_i_recons) -
                              np.mean(time_j_recons)],
                   'error_i': np.mean(error_i),
                   'error_j': np.mean(error_j),
                   'error_cross': np.abs(recons1 - recons2).mean()}
        np.save('/volatile/bsarthou/datas/wv_benchmark/loop_' +
                dic_wt_i['name']+'_VS_' +
                dic_wt_j['name']+'.npy', results)
        # print('DIFF', np.abs(recons1 - recons2).mean())
