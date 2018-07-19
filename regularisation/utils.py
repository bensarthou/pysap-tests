import numpy as np
import datetime

list_metric_names = ['snr', 'psnr', 'ssim', 'nrmse']


def log_tick_formatter(val, pos=None):
    return "{:.2e}".format(10**val)


def load_results(path, image=False, id_metric=2):
    list_dic = np.load(path)
    list_tau, list_decay = [], []
    res_metrics = [[] for i in range(len(list_metric_names))]
    std_metrics = [[] for i in range(len(list_metric_names))]
    min_metrics = [[] for i in range(len(list_metric_names))]
    max_metrics = [[] for i in range(len(list_metric_names))]

    # Load metrics for each set of sparkling parameters
    best_image = {'dic': {}, 'id_metric': id_metric, 'best_metric': 0.0,
                  'tau': 0.0, 'decay': 0.0}

    for dico in list_dic:
        list_tau.append(dico['tau'])
        list_decay.append(dico['decay'])

        # Check for the best reconstructed image in the list of dicts,
        # and its associated params.
        if (image and (dico['mean_'+list_metric_names[id_metric]] >
                       best_image['best_metric'])):

            best_image['best_metric'] = dico['mean_' +
                                             list_metric_names[id_metric]]
            best_image['tau'] = dico['tau']
            best_image['decay'] = dico['decay']
            best_image['data'] = dico['img_recons']

        stat_bool = True
        for j in range(len(list_metric_names)):
            res_metrics[j].append(dico['mean_'+list_metric_names[j]])
            try:
                std_metrics[j].append(dico['std_'+list_metric_names[j]])
                min_metrics[j].append(dico['min_'+list_metric_names[j]])
                max_metrics[j].append(dico['max_'+list_metric_names[j]])
            except:
                stat_bool = False
                pass

    if not stat_bool:
        print('No metrics statistics availables')
        res_metrics = [res_metrics]
    else:
        res_metrics = [res_metrics, std_metrics, min_metrics, max_metrics]

    if not image:
        return list_tau, list_decay, res_metrics
    else:
        return list_tau, list_decay, res_metrics, best_image
