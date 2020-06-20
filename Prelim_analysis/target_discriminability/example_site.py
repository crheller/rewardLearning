"""
Project target responses into TDR space. Compute state-dependent dprime.
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.baphy as nb
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import charlieTools.tuning_curves as tc
import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.reward_learning.behavior_helpers as bh

batch = 302
site = 'DRX006b.e1:64'  # DRX005c, DRX006b, DRX007a, DRX008b
fs = 40
time_window = 0.2   # n secs of evoked response to collapse over for calculating dprime
bins = int(fs * time_window)

options = {'batch': batch,
           'cellid': site,
           'rasterfs': fs,
           'pupil': True,
           'resp': True,
           'stim':False}

manager = BAPHYExperiment(batch=batch, siteid=site[:7])
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()

# extract appropriate shank
cells, _ = nb.parse_cellid(options)
rec['resp'] = rec['resp'].extract_channels(cells)
ncells = rec['resp'].shape[0]
rec = rec.and_mask(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL'], invert=True)

# extract target responses
targets = [e for e in rec.epochs.name.unique() if 'TAR_' in e]
rew_tar = bh.get_rewarded_targets(rec, manager)
nr_tar = bh.get_nonrewarded_targets(rec, manager)
R_center = []
rt = rec.copy()
rt.and_mask(['PreStimSilence'], invert=True)
for t in targets:
    _r = rec['resp'].extract_epoch(t, mask=rt['mask'])[:, :, :bins].mean(axis=-1, keepdims=True)
    _r -= _r.mean(axis=0, keepdims=True)
    R_center.append(_r)
R_center = np.concatenate(R_center, axis=0).transpose([1, 0, 2]).reshape(ncells, -1)

# get principal noise axis of target responses
pca = PCA(n_components=1)
pca.fit(R_center.T)

# combine rewarded targets to compute TDR for R vs. NR tones
rewarded = []
for t in rew_tar:
    _r = rec['resp'].extract_epoch(t, mask=rt['mask'])[:, :, :bins].mean(axis=-1, keepdims=True)
    rewarded.append(_r)
rewarded = np.concatenate(rewarded, axis=0).transpose([1, 0, 2]).reshape(ncells, -1)
not_rewarded = rec['resp'].extract_epoch(nr_tar[0], mask=rt['mask'])[:, :, :bins].mean(axis=-1, keepdims=True).squeeze().T

# compute TDR
tdr = dr.TDR(tdr2_init=pca.components_[[0]])
X = np.concatenate((rewarded, not_rewarded), axis=-1).T
y1 = np.concatenate((np.ones(rewarded.shape[1]), np.zeros(not_rewarded.shape[1])))
y2 = np.concatenate((np.zeros(rewarded.shape[1]), np.ones(not_rewarded.shape[1])))
y = np.stack([y1, y2])
tdr.fit(X, y.T)
tdr_weights = tdr.weights

# project target responses onto TDR space, compute dprime, plot, for each file.
rec['tdr1'] = rec['resp']._modified_copy(rec['resp']._data.T.dot(tdr_weights[[0], :].T).T)
rec['tdr2'] = rec['resp']._modified_copy(rec['resp']._data.T.dot(tdr_weights[[1], :].T).T)

files = [f for f in rec.epochs.name.unique() if 'FILE_' in f]
fig, ax = plt.subplots(1, len(files), figsize=(14, 4), sharex=True, sharey=True)
for i, f in enumerate(files):
    rt = rec.copy()
    rt = rt.and_mask([f])
    rt = rt.and_mask(['PreStimSilence'], invert=True)
    for t in targets:
        try:
            tdr1 = rec['tdr1'].extract_epoch(t, mask=rt['mask'], allow_incomplete=True)[:, :, :bins].mean(axis=-1)
            tdr2 = rec['tdr2'].extract_epoch(t, mask=rt['mask'], allow_incomplete=True)[:, :, :bins].mean(axis=-1)
            ax[i].scatter(tdr1, tdr2, s=15, label=t)
        except:
            ax[i].plot(np.nan, np.nan, label=t)
    ax[i].set_title(f)

fig.tight_layout()

plt.show()