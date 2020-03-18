"""
Analyze target response as function of time / reward size
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nems_lbhb.baphy as nb
from nems.recording import Recording
import preprocessing as prepoc
import plotting as cplt
import tuning_curves as tc
import copy

batch = 302
site = 'DRX008b'  # DRX005c, DRX006b, DRX007a, DRX008b
smooth = False
ylim = 0.23
rasterfs = 10

ops = {'batch': batch, 'siteid': site, 'rasterfs': rasterfs, 'pupil': 1, 'stim': 0, 'recache': False}
uri = nb.baphy_load_recording_uri(**ops)
rec = Recording.load(uri)
rec['resp'] = rec['resp'].rasterize()

r_passive = copy.deepcopy(rec)
r_passive = r_passive.and_mask(['PASSIVE_EXPERIMENT'])
r_passive = r_passive.apply_mask(reset_epochs=True)
ftc = tc.get_tuning_curves(r_passive)
bf = tc.get_bf(r_passive)
bf = bf[bf['sig']]
ftc = ftc.loc[pd.IndexSlice[:, bf[bf['sig']].index], :]

x = ftc.loc['r'].to_numpy(dtype=np.float)
idx = np.argsort(np.argmax(x, axis=1))
x = x[idx, :]
f, ax = plt.subplots(1, 1)
ax.imshow(x, aspect='auto', cmap='jet', interpolation='none')
ax.set_xticks(np.arange(0, x.shape[-1]))
ax.set_xticklabels([str(f) for f in ftc.loc['r'].columns], rotation=45, fontsize=8)
ax.set_title('all passive tuning, sig cells', fontsize=8)
ax.set_ylabel("Units, sorted by BF", fontsize=8)


files = [f for f in rec.epochs.name.unique() if 'FILE_' in f]
targets = [t for t in rec.epochs.name.unique() if 'TAR_' in t]
targets = np.sort(targets)
targets = [int(t.split('_')[1]) for t in targets]
nfiles = len(files) 

# for all cells, categorize based on their BF
# "ON TAR BF" = 0.5 octaves from mean rewarded tone
# "OFF TAR BF" = 0.5 octaves from mean non-rewarded tone
# "OFF BF" = everything else with sig tuning
tf = abs(targets[1] - targets[0]) > abs(targets[1] - targets[2])
if tf:
    rew_tar = np.mean(targets[1:])
    nw_tar = targets[0]
else:
    rew_tar = np.mean(targets[:2])
    nw_tar = targets[2]

bf['ON_TAR_BF'] = False
bf['OFF_TAR_BF'] = False
bf['OFF_BF'] = False

for c in bf.index:
    b = bf.loc[c]['BF']
    lt = rew_tar - (rew_tar / 4)
    ht = rew_tar + (rew_tar / 4)
    lnt = nw_tar - (nw_tar / 4)
    hnt = nw_tar + (nw_tar / 4)
    if (b >= lt) & ( b <= ht):
        bf.loc[c, 'ON_TAR_BF'] = True
    elif (b >= lnt) & ( b <= hnt):
        bf.loc[c, 'OFF_TAR_BF'] = True
    else:
        bf.loc[c, 'OFF_BF'] = True

ON_cells = bf[bf['ON_TAR_BF']].index 

for cellid in ON_cells:
    fig, ax = plt.subplots(1, nfiles, figsize=(12, 3), sharey=True)
    fig.canvas.set_window_title(cellid)
    for i, f in enumerate(files): 
        r = copy.deepcopy(rec)
        r = r.and_mask([f])
        if '_a_' in f:
            r = r.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])
        r = r.apply_mask(reset_epochs=True)

        ftc = tc.get_tuning_curves(r)
        vals = ftc.loc['r'].loc[cellid].to_numpy(dtype=np.float)
        sem = ftc.loc['sem'].loc[cellid].to_numpy(dtype=np.float)
        freqs = ftc.loc['r'].columns

        _bf = tc.get_bf(r)
        _bf = _bf.loc[cellid]['BF']

        ax[i].errorbar(freqs, vals, sem, color='k')
        ax[i].legend(['BF: {}'.format(_bf)])
        for tar in targets:
            ax[i].axvline(tar, color='r', linestyle='--')

        ax[i].set_title(f, fontsize=8)
        ax[i].set_xlabel('Freq.')
        ax[i].set_ylabel('Mean spk count')
        ax[i].set_xscale('log')

    fig.tight_layout()
    fig.savefig('/auto/users/hellerc/code/projects/rewardLearning/ON_TAR_FTC/{}.png'.format(cellid))


OFF_cells = bf[bf['OFF_TAR_BF']].index 

for cellid in OFF_cells:
    fig, ax = plt.subplots(1, nfiles, figsize=(12, 3), sharey=True)
    fig.canvas.set_window_title(cellid)
    for i, f in enumerate(files): 
        r = copy.deepcopy(rec)
        r = r.and_mask([f])
        if '_a_' in f:
            r = r.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])
        r = r.apply_mask(reset_epochs=True)

        ftc = tc.get_tuning_curves(r)
        vals = ftc.loc['r'].loc[cellid].to_numpy(dtype=np.float)
        sem = ftc.loc['sem'].loc[cellid].to_numpy(dtype=np.float)
        freqs = ftc.loc['r'].columns

        _bf = tc.get_bf(r)
        _bf = _bf.loc[cellid]['BF']

        ax[i].errorbar(freqs, vals, sem, color='k')
        ax[i].legend(['BF: {}'.format(_bf)])
        for tar in targets:
            ax[i].axvline(tar, color='r', linestyle='--')

        ax[i].set_title(f, fontsize=8)
        ax[i].set_xlabel('Freq.')
        ax[i].set_ylabel('Mean spk count')
        ax[i].set_xscale('log')

    fig.tight_layout()
    fig.savefig('/auto/users/hellerc/code/projects/rewardLearning/OFF_TAR_FTC/{}.png'.format(cellid))

plt.show()
OFF_TAR_cells = bf[bf['OFF_BF']].index 

for cellid in OFF_TAR_cells:
    fig, ax = plt.subplots(1, nfiles, figsize=(12, 3), sharey=True)
    fig.canvas.set_window_title(cellid)
    for i, f in enumerate(files): 
        r = copy.deepcopy(rec)
        r = r.and_mask([f])
        if '_a_' in f:
            r = r.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])
        r = r.apply_mask(reset_epochs=True)

        ftc = tc.get_tuning_curves(r)
        vals = ftc.loc['r'].loc[cellid].to_numpy(dtype=np.float)
        sem = ftc.loc['sem'].loc[cellid].to_numpy(dtype=np.float)
        freqs = ftc.loc['r'].columns

        _bf = tc.get_bf(r)
        _bf = _bf.loc[cellid]['BF']

        ax[i].errorbar(freqs, vals, sem, color='k')
        ax[i].legend(['BF: {}'.format(_bf)])
        for tar in targets:
            ax[i].axvline(tar, color='r', linestyle='--')

        ax[i].set_title(f, fontsize=8)
        ax[i].set_xlabel('Freq.')
        ax[i].set_ylabel('Mean spk count')
        ax[i].set_xscale('log')

    fig.tight_layout()
    fig.savefig('/auto/users/hellerc/code/projects/rewardLearning/OFF_FTC/{}.png'.format(cellid))

plt.close('all')