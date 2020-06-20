"""
Compute first PC of rewarded target response, first PC of n.r. target response.
Compute state / condition-dependent tuning curve for each axis.
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import charlieTools.tuning_curves as tc
import charlieTools.reward_learning.behavior_helpers as bh

batch = 302
site = 'DRX006b'  # DRX005c, DRX006b, DRX007a, DRX008b
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

# z-score response of all neurons
r = rec['resp']._data 
rz = r - r.mean(axis=-1, keepdims=True)
rz = rz / rz.std(axis=-1, keepdims=True)
rec['resp'] = rec['resp']._modified_copy(rz)

# mask out all trials except for early trials / FA trials (where there may be nans)
rec = rec.and_mask(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL'], invert=True)

targets = [e for e in rec.epochs.name.unique() if 'TAR_' in e]
rew_tar = bh.get_rewarded_targets(rec, manager)
nr_tar = bh.get_nonrewarded_targets(rec, manager)

# get rewarded target population response
rewarded = []
for t in rew_tar:
    _r = rec['resp'].extract_epoch(str(t), mask=rec['mask']).mean(axis=0)
    rewarded.append(_r)

rewarded = np.stack(rewarded).transpose([1, 0, 2]).reshape(ncells, -1)
pca = PCA(n_components=2)
pca.fit(rewarded.T)
rec['rewarded'] = rec['resp']._modified_copy(np.matmul(rec['resp'].as_continuous().T, pca.components_[[0]].T).T)

# get non-rewarded target population response
not_rewarded = []
for t in nr_tar:
    _r = rec['resp'].extract_epoch(str(t), mask=rec['mask']).mean(axis=0)
    not_rewarded.append(_r)

not_rewarded = np.stack(not_rewarded).transpose([1, 0, 2]).reshape(ncells, -1)
pca = PCA(n_components=2)
pca.fit(not_rewarded.T)
rec['not_rewarded'] = rec['resp']._modified_copy(np.matmul(rec['resp'].as_continuous().T, pca.components_[[0]].T).T)

# replace rec with pop signals
rpop = rec.copy()
rpop.create_mask(True)
rpop['resp'] = rec['resp']._modified_copy(np.concatenate((rec['rewarded']._data, rec['not_rewarded']._data), axis=0))
rpop['resp'].chans = ['rewarded', 'not_rewarded']

# plot tuning per file
rew = np.mean([int(f.replace('TAR_', '')) for f in rew_tar])
nr = np.mean([int(f.replace('TAR_', '')) for f in nr_tar])
files = [f for f in rec.epochs.name.unique() if 'FILE' in f]
files = [f for f in files if f!='FILE_DRX005c07_p_BVT']
fig, ax = plt.subplots(1, len(files), figsize=(16, np.ceil(12 / len(files))+1), sharey=True)
for i, f in enumerate(files):
    r = rpop.copy()
    r = r.and_mask([f])
    if '_a_' in f:
        # correct FTC
        correct = r.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])
        ftc_correct = tc.get_tuning_curves(correct)

        # incorrect FTC
        incorrect = r.and_mask(['MISS_TRIAL', 'INCORRECT_HIT_TRIAL'])
        ftc_incorrect = tc.get_tuning_curves(incorrect)

        ax[i].plot(ftc_correct.columns, ftc_correct.loc['r', 'rewarded'], label='rewarded PC1')
        a = (ftc_correct.loc['r', 'rewarded'].values-ftc_correct.loc['sem', 'rewarded'].values).astype(np.float)
        b = (ftc_correct.loc['r', 'rewarded'].values+ftc_correct.loc['sem', 'rewarded'].values).astype(np.float)
        ax[i].fill_between(ftc_correct.columns, a, b, alpha=0.3)

        ax[i].plot(ftc_correct.columns, ftc_correct.loc['r', 'not_rewarded'], label='not rewarded PC1')
        a = (ftc_correct.loc['r', 'not_rewarded'].values-ftc_correct.loc['sem', 'not_rewarded'].values).astype(np.float)
        b = (ftc_correct.loc['r', 'not_rewarded'].values+ftc_correct.loc['sem', 'not_rewarded'].values).astype(np.float)
        ax[i].fill_between(ftc_correct.columns, a, b, alpha=0.3)

        ax[i].set_title('Active')

    else:
        # passive FTC
        passive = r.and_mask(['PASSIVE_EXPERIMENT'])
        ftc_passive = tc.get_tuning_curves(passive)

        ax[i].plot(ftc_passive.columns, ftc_passive.loc['r', 'rewarded'], label='rewarded PC1')
        a = (ftc_passive.loc['r', 'rewarded'].values-ftc_passive.loc['sem', 'rewarded'].values).astype(np.float)
        b = (ftc_passive.loc['r', 'rewarded'].values+ftc_passive.loc['sem', 'rewarded'].values).astype(np.float)
        ax[i].fill_between(ftc_passive.columns, a, b, alpha=0.3)
        
        ax[i].plot(ftc_passive.columns, ftc_passive.loc['r', 'not_rewarded'], label='not rewarded PC1')
        a = (ftc_passive.loc['r', 'not_rewarded'].values-ftc_passive.loc['sem', 'not_rewarded'].values).astype(np.float)
        b = (ftc_passive.loc['r', 'not_rewarded'].values+ftc_passive.loc['sem', 'not_rewarded'].values).astype(np.float)
        ax[i].fill_between(ftc_passive.columns, a, b, alpha=0.3)
        ax[i].set_title("Passive")

    ax[i].axvline(rew, color='green', lw=3, linestyle='--')
    ax[i].axvline(nr, color='red', lw=3, linestyle='--')

ax[0].legend(frameon=False)

fig.tight_layout()

plt.show()
