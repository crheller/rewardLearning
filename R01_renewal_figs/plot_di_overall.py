import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update({'svg.fonttype': 'none'})

# set up axes
plt.figure(figsize=(14, 4))
bar_sem_ax = plt.subplot2grid((1, 6), (0, 0), colspan=1)
bar_scatter_ax = plt.subplot2grid((1, 6), (0, 1), colspan=1)
boxplot_ax = plt.subplot2grid((1, 6), (0, 2), colspan=1)


di_metric = 'ALL_DI' # R_DI
# load data
crd = pd.read_pickle('/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/results/Cordyceps_overallStats.pickle')
drx = pd.read_pickle('/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/results/Drechsler_overallStats.pickle')

# get stats
crd_mean = np.round(crd[di_metric].mean(), 3)
drx_mean = np.round(drx[di_metric].mean(), 3)
crd_pval = np.round(ss.wilcoxon(crd[di_metric]-0.5)[1], 3)
drx_pval = np.round(ss.wilcoxon(drx[di_metric]-0.5)[1], 3)

# plot data

# bar plot +/- sem
bar_sem_ax.bar([0, 1], [crd[di_metric].mean(), drx[di_metric].mean()], 
                        yerr=[crd[di_metric].sem(), drx[di_metric].sem()],
                        edgecolor='k', color='none', lw=2)
bar_sem_ax.set_ylabel(di_metric)
bar_sem_ax.set_xticks([0, 1])
bar_sem_ax.set_xticklabels(['CRD', 'DRX'])
bar_sem_ax.set_ylim((0, 1.1))
bar_sem_ax.set_title('CRD: {0}, pval: {1} \n DRX: {2}, pval: {3}'.format(crd_mean,
                                                                    crd_pval,
                                                                    drx_mean,
                                                                    drx_pval))

# bar plot with individual data points
bar_scatter_ax.bar([0, 1], [crd[di_metric].mean(), drx[di_metric].mean()],
                        edgecolor='k', color='none', lw=2)
crd_idx = np.random.normal(0, 0.06, crd.shape[0])
drx_idx = np.random.normal(1, 0.06, drx.shape[0])
bar_scatter_ax.plot(crd_idx, crd[di_metric], 'k.')
bar_scatter_ax.plot(drx_idx, drx[di_metric], 'k.')
bar_scatter_ax.set_ylabel(di_metric)
bar_scatter_ax.set_xticks([0, 1])
bar_scatter_ax.set_xticklabels(['CRD', 'DRX'])
bar_scatter_ax.set_ylim((0, 1.1))


# boxplot
box = boxplot_ax.boxplot([crd[di_metric], drx[di_metric]], positions=[0, 1])
for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box[item], color='k')
boxplot_ax.set_ylabel(di_metric)
boxplot_ax.set_xticks([0, 1])
boxplot_ax.set_xticklabels(['CRD', 'DRX'])
boxplot_ax.set_ylim((0, 1.1))


plt.tight_layout()

plt.show()

