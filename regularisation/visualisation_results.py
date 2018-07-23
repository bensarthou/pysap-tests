import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import datetime

from utils import *

id_metrics = 2  # SSIM

print('LOADING DATA')
directory = '/volatile/bsarthou/datas/sparkling/stat_sparkling_'


path_res = ['1e-08_201869_031.npy',
            '1e-07_201869_30.npy',
            '1e-06_201869_1115.npy',
            '1e-05_201869_1936.npy',
            '0.0001_2018610_417.npy',
            '0.1_2018613_950.npy',
            '0.1_2018625_1621.npy',
            '0.1_2018629_2314.npy',
            '0.1_201879_167.npy',
            '0.0_2018613_135.npy',
            '0.0_2018625_2214.npy',
            '0.0_2018630_758.npy',
            '0.0_201879_224.npy',
            '10.0_2018614_142.npy',
            '100.0_2018616_154.npy',
            '100.0_2018626_336.npy',
            '100.0_2018630_1635.npy',
            '100.0_2018710_247.npy',
            '200.0_2018616_945.npy',
            '200.0_2018626_919.npy',
            '200.0_201871_034.npy',
            '200.0_2018710_754.npy',
            '250.0_2018616_1655.npy',
            '250.0_2018626_1542.npy',
            '250.0_201871_810.npy',
            '250.0_2018710_1344.npy'
            '300.0_2018616_2344.npy',
            '300.0_2018626_2158.npy',
            '300.0_201871_1514.npy',
            '300.0_2018710_1949.npy',
            '500.0_2018617_69.npy',
            '500.0_2018627_446.npy',
            '500.0_201871_2118.npy',
            '500.0_2018711_244.npy',
            '1000.0_2018617_1336.npy',
            '1000.0_2018627_1228.npy',
            '1000.0_201872_411.npy',
            '1000.0_2018711_1029.npy',
            '2000.0_2018617_2154.npy']

# For slides:
path_res = [
            '0.1_2018613_950.npy',
            '0.1_2018625_1621.npy',
            '0.1_2018629_2314.npy',
            '0.1_201879_167.npy',
            '0.0_2018613_135.npy',
            '0.0_2018625_2214.npy',
            '0.0_2018630_758.npy',
            '0.0_201879_224.npy',
            '100.0_2018616_154.npy',
            '100.0_2018626_336.npy',
            '100.0_2018630_1635.npy',
            '100.0_2018710_247.npy',
            '200.0_2018616_945.npy',
            '200.0_2018626_919.npy',
            '200.0_201871_034.npy',
            '200.0_2018710_754.npy',
            '250.0_2018616_1655.npy',
            '250.0_2018626_1542.npy',
            '250.0_201871_810.npy',
            '250.0_2018710_1344.npy',
            '300.0_2018616_2344.npy',
            '300.0_2018626_2158.npy',
            '300.0_201871_1514.npy',
            '300.0_2018710_1949.npy',
            '500.0_2018617_69.npy',
            '500.0_2018627_446.npy',
            '500.0_201871_2118.npy',
            '500.0_2018711_244.npy',
            '1000.0_2018617_1336.npy',
            '1000.0_2018627_1228.npy',
            '1000.0_201872_411.npy',
            '1000.0_2018711_1029.npy']

# path_res = ['100.0_2018626_336.npy']
# path_res = ['100.0_2018616_154.npy']
# path_res = ['100.0_2018626_336.npy', '100.0_2018616_154.npy']

nb_lambdas = len(path_res)

titles = [float(p.split('_')[0]) for p in path_res]
# WARNING must be same size as path_res
colormap = ['magma', 'seismic', 'rainbow', 'gray', 'plasma']
colormap2 = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
list_metric_names = ['snr', 'psnr', 'ssim', 'nrmse']

#############################################################################
# CREATING DATA CUBE
# ------------------
#
# Load all the dictionnaries listed in path_res and extract metrics data in a
# cube for data visualisation


# Creating a table with columns: lambda, tau, decay, mean_metric, std_metric,
# min, max

for (p, path) in enumerate(path_res):
    # Loading, for a lambda, the sparkling parameters and metrics associated
    tau, decay, metrics = load_results(directory + path)
    # Getting lambda from namefile
    lbda = float(path.split('_')[0])
    # Creating
    temp_table = np.zeros((len(decay), 7))
    temp_table[:, 0] = lbda*np.ones((len(decay),))
    temp_table[:, 1] = tau
    temp_table[:, 2] = decay
    temp_table[:, 3] = metrics[0][id_metrics]

    try:
        temp_table[:, 4] = metrics[1][id_metrics]
        temp_table[:, 5] = metrics[2][id_metrics]
        temp_table[:, 6] = metrics[3][id_metrics]
    except IndexError:
        pass

    if(p == 0):
        X = temp_table
    else:
        X = np.append(X, temp_table, axis=0)

    print('{} out of {} loaded'.format(p+1, len(path_res)), end="\r")

# Rearranging the table to do a 3D cube of the data, each axis being a variable
lambdas = np.unique(X[:, 0])
taus = np.unique(X[:, 1])
decays = np.unique(X[:, 2])

cube_mean = np.zeros((len(taus), len(decays), len(lambdas)))
cube_std = np.zeros((len(taus), len(decays), len(lambdas)))
cube_min = np.zeros((len(taus), len(decays), len(lambdas)))
cube_max = np.zeros((len(taus), len(decays), len(lambdas)))

for (i, t) in enumerate(taus):
    for (j, d) in enumerate(decays):
        for (k, l) in enumerate(lambdas):
            selector = np.logical_and(X[:, 0] == l,
                                      np.logical_and(X[:, 1] == t,
                                                     X[:, 2] == d))

            if(np.any(selector)):
                # If the set of tau, decay, lambda exist in the dataset
                cube_mean[i, j, k] = X[selector][0, 3]
                cube_std[i, j, k] = X[selector][0, 4]
                cube_min[i, j, k] = X[selector][0, 5]
                cube_max[i, j, k] = X[selector][0, 6]
            else:
                cube_mean[i, j, k] = np.nan
                cube_std[i, j, k] = np.nan
                cube_min[i, j, k] = np.nan
                cube_max[i, j, k] = np.nan


#############################################################################
# PLOT LAMBDA SURFACES
# --------------------
#
# For each lambda (in each dictionary), plot the value of all metrics according
# to tau and decay sparkling parameters. Each surface is a specific lambda

fig = plt.figure()
axJ = fig.add_subplot(1, 1, 1, projection='3d')
axJ.set_title('lambdas:' + str(lambdas))
axJ.set_xlabel('tau')
axJ.set_ylabel('decay')
for i in range(cube_mean.shape[2]):
    U, V = np.meshgrid(taus, decays)
    surf = axJ.plot_wireframe(U, V, cube_mean[:, :, i].transpose(),
                              color=colormap2[i % (len(colormap2))],
                              label='lambda {}'.format(lambdas[i]))
    # surf._facecolors2d = surf._facecolors3d
    # surf._edgecolors2d = surf._edgecolors3d
    leg = axJ.legend()
    LH = leg.legendHandles
    LH[i].set_color(colormap2[i % (len(colormap2))])

# plt.show()

print('LAMBDA, TAU OR LAMDA, DECAY FIXED')
fig5 = plt.figure()
fig5.suptitle('tau, decay, lambda fixed')

nb_samples = 40  # WARNING hardcoded value
value_fixed_tau = 0.4
value_fixed_decay = 5
value_fixed_lambda = 100

values_fixed = [value_fixed_tau, value_fixed_decay, value_fixed_lambda]

dim = ['tau', 'decay', 'lambda']

ax_choice = [{'fix': 1, 'plot': 0, 'stack': 2},
             {'fix': 0, 'plot': 1, 'stack': 2},
             {'fix': 1, 'plot': 2, 'stack': 0},
             {'fix': 0, 'plot': 2, 'stack': 1}]

tdl = [taus, decays, lambdas]
for (i, axes) in enumerate(ax_choice):
    # Select the surface for the fixed value
    fx_mean = np.take(cube_mean,
                      np.where(tdl[axes['fix']] == values_fixed[axes['fix']]),
                      axis=axes['fix'])
    fx_std = np.take(cube_std,
                     np.where(tdl[axes['fix']] == values_fixed[axes['fix']]),
                     axis=axes['fix'])
    fx_min = np.take(cube_min,
                     np.where(tdl[axes['fix']] == values_fixed[axes['fix']]),
                     axis=axes['fix'])
    fx_max = np.take(cube_max,
                     np.where(tdl[axes['fix']] == values_fixed[axes['fix']]),
                     axis=axes['fix'])
    fx_mean, fx_std = np.squeeze(fx_mean), np.squeeze(fx_std)
    fx_min, fx_max = np.squeeze(fx_min), np.squeeze(fx_max)

    fx_value = values_fixed[axes['fix']]
    x = tdl[axes['plot']]
    labels = tdl[axes['stack']]
    # For specific cases, reorder the selected surface to have the plot
    # variable as x axis and the stack variable as the y axis

    if(axes['plot'] > axes['stack']):
        fx_mean = fx_mean.transpose()
        fx_std = fx_std.transpose()
        fx_min = fx_min.transpose()
        fx_max = fx_max.transpose()

    # Create the plot
    axI = fig5.add_subplot(2, np.ceil(len(ax_choice)/2), i+1)
    axI.set_title('{} along {},'
                  '{} fixed at {}'.format(list_metric_names[id_metrics],
                                          dim[axes['plot']],
                                          dim[axes['fix']],
                                          fx_value))

    axI.set_xlabel(dim[axes['plot']])
    axI.set_ylabel(list_metric_names[id_metrics])

    for j in range(cube_mean.shape[axes['stack']]):

        y_mean = fx_mean[:, j]
        y_std = fx_std[:, j]
        y_min = fx_min[:, j]
        y_max = fx_max[:, j]

        axI.plot(x, y_mean,
                 color=colormap2[j % (len(colormap2))],
                 label=str(dim[axes['stack']])+': '+str(labels[j]))

        # axI.plot(x,
        #          y_mean - y_std/np.sqrt(nb_samples),
        #          color=colormap2[j % (len(colormap2))],
        #          # label=str(labels[j])+' std-',
        #          linestyle='--')
        #
        # axI.plot(x,
        #          y_mean + y_std/np.sqrt(nb_samples),
        #          color=colormap2[j % (len(colormap2))],
        #          # label=str(labels[j])+' std+',
        #          linestyle='--')

        axI.fill_between(x,
                         y_mean - y_std/np.sqrt(nb_samples),
                         y_mean + y_std/np.sqrt(nb_samples),
                         color=colormap2[j % (len(colormap2))],
                         # label=str(labels[j])+' std',
                         linestyle='None',
                         alpha=0.3)

        # axI.plot(x,
        #          y_min,
        #          color=colormap2[j % (len(colormap2))],
        #          label=str(labels[j])+' min',
        #          linestyle=':')
        #
        # axI.plot(x,
        #          y_max,
        #          color=colormap2[j % (len(colormap2))],
        #          label=str(labels[j])+' max',
        #          linestyle=':')

    axI.legend()
plt.show()
