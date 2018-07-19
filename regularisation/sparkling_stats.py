from scipy.io import loadmat
import pandas as pd
import numpy as np
from modopt.math.metrics import ssim, snr, psnr, nrmse
from wrapper_pysap import recons_pysap, recons_pysap_3D
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
import datetime
import os
import sys


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


#############################################################################
# Loading the train and test image datasets
# -----------------------------------------
#
# We will train our models on a set of images, and also test reconstruction on
# another one. Output of this section must be two list of np.arrays, 2D or 3D
print('LOADING IMAGES')
images_directory = '/volatile/bsarthou/datas/FLASH3D/'
train_set = pd.HDFStore(images_directory + 'training_data.h5')['df']
test_set = pd.HDFStore(images_directory + 'testing_data.h5')['df']
X_train = np.dstack(train_set['data'])
X_test = np.dstack(test_set['data'])

pad = np.zeros((30, 384, 400))

X_train = np.concatenate((pad, X_train, pad), axis=0)

# # Reshape for squared images (256*256)
# X_train = X_train[((X_train.shape[0]//2)-128):((X_train.shape[0]//2)+128),
#                   ((X_train.shape[1]//2)-128):((X_train.shape[1]//2)+128), :]
#
# X_test = X_test[((X_test.shape[0]//2)-128):((X_test.shape[0]//2)+128),
#                 ((X_test.shape[1]//2)-128):((X_test.shape[1]//2)+128), :]

X_train, X_test = X_train.transpose(), X_test.transpose()

shape_img = (X_train[0].shape)

print('Number of images in the db:', X_train.shape[0])
X_train = X_train[:40, :, :]


#############################################################################
# Reconstruction algorithm
# ---------------------------------------------
#
# We need to fix a CS reconstruction algorithm which will help determine which
# sampling pattern is better. The wrapper around the reconstruction algorithm
# must be as such: recons(image, samples, opt),with image a 2D or 3D np.array,
# samples the locations in [-0.5, 0.5[, and opt a dictionnary with the others
# parameters of the algorithm.

mu = float(sys.argv[1])

# g = recons_pysap
# kwargs = {'wavelet_name': "UndecimatedBiOrthogonalTransform",
#           'nb_scales': 4,
#           'std_est': None,
#           'std_est_method': None,
#           'std_thr': 2.,
#           'mu': mu,
#           'tau': None,
#           'sigma': None,
#           'relaxation_factor': 1.0,
#           'nb_of_reweights': 2,
#           'max_nb_of_iter': 100,
#           'add_positivity': False,
#           'atol': 1e-4,
#           'non_cartesian': True,
#           'uniform_data_shape': shape_img,
#           'verbose': 0}

g = recons_pysap_3D
kwargs = {'std_est': None,
          'std_est_method': None,
          'std_thr': 2.,
          'mu': mu,
          'tau': None,
          'sigma': None,
          'relaxation_factor': 1.0,
          'nb_of_reweights': 2,
          'max_nb_of_iter': 100,
          'add_positivity': False,
          'atol': 1e-4,
          'verbose': 0,
          'get_cost': True}

#############################################################################
# Loading the sampling patterns we want to test
# ---------------------------------------------
#
# This algorithm explores a discrete sampling space,in which all binary mask
# will be loaded in masks list.
print('LOADING SAMPLES')
# samples_dirpath = '/home/bs255482/.local/share/pysap/'
# samples_paths = ['samples_radial_x15_34x3072.mat',
#                  'samples_sparkling_x15_34x3072.mat',
#                  'samples_radial_x8_64x3072.mat',
#                  'samples_sparkling_x8_64x3072.mat']

samples_dirpath = '/volatile/bsarthou/datas/sparkling/samples_sparkling_/' + \
                  'N384_radsym_nc24/ns3072/'
samples_paths = []
# base_dir = '2018-Jun-06_N384_nc24_ns3072_Dt0.01ms_OS1_decim1_'

base_dir = '2018-Jul-06_N384_nc24_ns3072_Dt0.01ms_OS1_decim1_'

list_res = []

decays = [1, 1.75, 2.5, 3.25, 4]
taus = [0.1, 0.2, 0.3, 0.4, 0.5]

# decays = [1.75, 2.5]
# taus = [0.7, 0.9]

cnt = 0
for decay in decays:
    for tau in taus:

        res_dict = {'shape_img': shape_img,
                    'mu': mu,
                    'nc': 24,
                    'ns': 3072,
                    'decim': 1,
                    'decay': decay,
                    'tau': tau}

        path = base_dir + 'decay{}_tau{}/'.format(decay, tau) + \
                          'Samples_SPARKLING_N384_R2_nc24x3072_Dt10us.mat'

        mask = loadmat(samples_dirpath + path)
        mask = mask['samples']

        # Normalisation between [-0.5, 0.5[
        mask = mask/(2*np.abs(mask).max())

        # means of metrics
        gamma_temp = [[], [], [], []]

        for j in range(X_train.shape[0]):
            image = X_train[j]
            img_recons, cost = g(image, mask, **kwargs)
            m = np.array([snr(image, img_recons),
                          psnr(image, img_recons),
                          ssim(image, img_recons, None),
                          nrmse(image, img_recons)])

            print('ID img, decay, tau, recons metric: ({}, {}, {},'
                  '{})\n'.format(j, decay, tau, m))
            # gamma_temp = gamma_temp + m
            for i in range(len(m)):
                gamma_temp[i].append(m[i])

            cnt += 1
            print('PROGRESSION: {:4f}%\n'.format(
                                (cnt*100) /
                                (len(decays)*len(taus)*X_train.shape[0])))

        # Save metrics and data of the reconstructions
        res_dict['mean_snr'] = np.mean(gamma_temp[0])
        res_dict['mean_psnr'] = np.mean(gamma_temp[1])
        res_dict['mean_ssim'] = np.mean(gamma_temp[2])
        res_dict['mean_nrmse'] = np.mean(gamma_temp[3])

        res_dict['std_snr'] = np.std(gamma_temp[0])
        res_dict['std_psnr'] = np.std(gamma_temp[1])
        res_dict['std_ssim'] = np.std(gamma_temp[2])
        res_dict['std_nrmse'] = np.std(gamma_temp[3])

        res_dict['min_snr'] = np.min(gamma_temp[0])
        res_dict['min_psnr'] = np.min(gamma_temp[1])
        res_dict['min_ssim'] = np.min(gamma_temp[2])
        res_dict['min_nrmse'] = np.min(gamma_temp[3])

        res_dict['max_snr'] = np.max(gamma_temp[0])
        res_dict['max_psnr'] = np.max(gamma_temp[1])
        res_dict['max_ssim'] = np.max(gamma_temp[2])
        res_dict['max_nrmse'] = np.max(gamma_temp[3])

        res_dict['img_orig'] = np.copy(image)
        res_dict['img_recons'] = np.copy(img_recons)
        res_dict['cost'] = np.copy(cost)

        res_dict['nb_decays'] = len(decays)
        res_dict['nb_taus'] = len(taus)

        list_res.append(res_dict)

date = datetime.datetime.now()
namefile = 'stat_sparkling_{}_{}{}{}_{}{}.npy'.format(mu, date.year,
                                                      date.month,
                                                      date.day, date.hour,
                                                      date.minute)

save_dir = '/volatile/bsarthou/datas/sparkling/'

np.save(save_dir + namefile, list_res)
