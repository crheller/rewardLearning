"""
Use dPCA to analyze target responses

TODO: Need to split up response matrix into another dimension. Need dims stim x time x outcome x neuron
""" 

from dPCA.dPCA import dPCA

from nems_lbhb.baphy_experiment import BAPHYExperiment
from sklearn.decomposition import PCA
import scipy.ndimage.filters as sf
import numpy as np
import matplotlib.pyplot as plt
import charlieTools.tuning_curves as tc
import charlieTools.reward_learning.behavior_helpers as bh

figpath = '/home/charlie/Desktop/lbhb/code/projects/rewardLearningFigs/'

batch = 302
site = 'DRX008b'  # DRX005c, DRX006b, DRX007a, DRX008b
fs = 40

options = {'batch': batch,
           'cellid': site,
           'rasterfs': fs,
           'pupil': True,
           'resp': True,
           'stim':False}

manager = BAPHYExperiment(batch=batch, siteid=site)
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()
ncells = rec['resp'].shape[0]

# only include TARGET response data
rec = rec.and_mask(['TARGET'])
rec = rec.apply_mask(reset_epochs=True)

# z-score response of all neurons
r = rec['resp']._data 
rz = r - r.mean(axis=-1, keepdims=True)
rz = rz / rz.std(axis=-1, keepdims=True)
rec['resp'] = rec['resp']._modified_copy(rz)

# Pull out the four response matrices for each behavior outcome

# HIT
hit = rec['resp'].extract_epoch('HIT_TRIAL')
# MISS
try:
    miss = rec['resp'].extract_epoch('MISS_TRIAL')
except:
    miss = None
# CORR. REJECT
cmiss = rec['resp'].extract_epoch('CORRECT_REJECT_TRIAL')
# INCORR. HIT
ihit = rec['resp'].extract_epoch('INCORRECT_HIT_TRIAL')

# define low-D projection
nr_resp = np.stack((cmiss.mean(axis=0), ihit.mean(axis=0)))
r_resp = hit.mean(axis=0)
if miss is not None:
    r_resp = np.stack((r_resp, miss.mean(axis=0)))

all_resp = np.stack((r_resp, nr_resp)).transpose([2, -1, 0, 1])

#dpca = dPCA(n_components=5, labels='ts', join={'ts_join': ['s', 'ts']})
dpca = dPCA(n_components=5, labels='tsc')
_ = dpca.fit_transform(all_resp)
var_exp_t = dpca.explained_variance_ratio_['t']
var_exp_ts = dpca.explained_variance_ratio_['ts']
var_exp_tc = dpca.explained_variance_ratio_['tc']
resp = [hit, miss, cmiss, ihit]
color = ['darkorange', 'darkorange', 'darkmagenta', 'darkmagenta']
linestyle = ['-', '--', '-', '--']
label = ['HIT', 'MISS', 'Corr. Rej.', 'Incorr. hit']
f, ax = plt.subplots(3, 4, figsize=(12, 9), sharey=True)
for i in range(0, 4):
    for r, c, ls, l in zip(resp, color, linestyle, label):
        if r is not None:
            proj = dpca.transform(r.transpose([1, 2, 0]))['t'][i, :, :]
            projm = sf.gaussian_filter1d(proj.mean(axis=-1), sigma=1.2)

            ax[0, i].plot(projm, color=c, label=l, linestyle=ls)
            ax[0, i].set_title('dPCA time comp. {0}, \n var explained: {1}'.format(i, round(var_exp_t[i], 3)))
            ax[0, i].axvline(int(0.1 * fs), linestyle='--', color='grey')
            ax[0, i].axvline(int(0.6 * fs), linestyle='--', color='grey')

            proj = dpca.transform(r.transpose([1, 2, 0]))['ts'][i, :, :]
            projm = sf.gaussian_filter1d(proj.mean(axis=-1), sigma=1.2)

            ax[1, i].plot(projm, color=c, label=l, linestyle=ls)
            ax[1, i].set_title('dPCA stimulus comp. {0}, \n var explained: {1}'.format(i, round(var_exp_ts[i], 3)))
            ax[1, i].axvline(int(0.1 * fs), linestyle='--', color='grey')
            ax[1, i].axvline(int(0.6 * fs), linestyle='--', color='grey')

            proj = dpca.transform(r.transpose([1, 2, 0]))['tc'][i, :, :]
            projm = sf.gaussian_filter1d(proj.mean(axis=-1), sigma=1.2)

            ax[2, i].plot(projm, color=c, label=l, linestyle=ls)
            ax[2, i].set_title('dPCA behavior comp. {0}, \n var explained: {1}'.format(i, round(var_exp_tc[i], 3)))
            ax[2, i].axvline(int(0.1 * fs), linestyle='--', color='grey')
            ax[2, i].axvline(int(0.6 * fs), linestyle='--', color='grey')

ax[0, 0].legend(frameon=False)
f.tight_layout()
f.canvas.set_window_title(site)

f.savefig(figpath+'dPCA_{}.png'.format(site))

plt.show()