"""
Analyze target response as function of time / reward size
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sf

import nems_lbhb.baphy as nb
from nems.recording import Recording
import preprocessing as prepoc
import plotting as cplt
import copy

batch = 302
site = 'DRX008b'  # DRX005c, DRX006b, DRX007a, DRX008b
smooth = False
ylim = 0.23
rasterfs = 10

ops = {'batch': batch, 'siteid': site, 'rasterfs': rasterfs, 'pupil': 1, 'stim': 0, 'recache': True}
uri = nb.baphy_load_recording_uri(**ops)
rec = Recording.load(uri)
rec['resp'] = rec['resp'].rasterize()

norm = (1 / rec['resp']._data.max(axis=-1, keepdims=True)) * rec['resp']._data
rec['norm'] = rec['resp']._modified_copy(norm)

# smooth data
if smooth:
    smooth = sf.gaussian_filter1d(rec['norm']._data, 1)
    rec['norm'] = rec['norm']._modified_copy(smooth)

targets = [t for t in rec.epochs.name.unique() if 'TAR_' in t]
files = [f for f in rec.epochs.name.unique() if 'FILE_' in f]

# figure out trial timing
all_tar_resp = rec['resp'].extract_epoch('TARGET')
time = np.arange(0, 0.7 + (1 / rasterfs), 1 / rasterfs)
idx = np.arange(0, time[-1] * rasterfs, dtype=int)
onset = 0.1
offset = 0.6

fig, ax = plt.subplots(1, len(files), figsize=(14, 4))

for i, f in enumerate(files):

    r = copy.deepcopy(rec)
    r = r.and_mask([f])
    if '_a_' in f:
        #r = r.and_mask(['HIT_TRIAL', 'INCORRECT_HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL'])
        r = r.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])
    r = r.apply_mask(reset_epochs=True)

    # normalize psth
    r['resp'] = r['norm']
    r = prepoc.generate_psth(r)
    r['psth_norm'] = r['psth']
    pop_psth = np.nanmean(r['psth_norm']._data, axis=0, keepdims=True)
    r['pop_psth'] = r['psth']._modified_copy(pop_psth)
    pop_psth_sem = np.nanstd(r['psth_norm']._data, axis=0, keepdims=True) / np.sqrt(r['resp'].shape[0])
    r['pop_psth_sem'] = r['psth']._modified_copy(pop_psth_sem)

    ax[i].set_title(f, fontsize=8)

    for tar in targets:
        try:
            tar_resp = r['pop_psth'].extract_epoch(tar)[0, :, idx].squeeze()
            tar_sem = r['pop_psth_sem'].extract_epoch(tar)[0, :, idx].squeeze()
            ax[i].plot(time, tar_resp, label=tar)
            ax[i].fill_between(time, tar_resp-tar_sem, tar_resp+tar_sem, alpha=0.5, lw=0)
        except:
            ax[i].plot(time, np.nan * np.ones(len(time)))
            ax[i].fill_between(time, np.nan * np.ones(len(time)), np.nan * np.ones(len(time)), alpha=0.5, lw=0)
    
    # figure out onset / offset bin
    ax[i].axvline(onset, color='lightgrey', linestyle='--')
    ax[i].axvline(offset, color='lightgrey', linestyle='--')

    # add plot labels set lims
    ax[i].legend(fontsize=8)
    ax[i].set_xlabel('Time (s)', fontsize=8)
    if i == 0:
        ax[i].set_ylabel('Norm. Response')
    ax[i].set_ylim((0, ylim))
    ax[i].set_aspect(cplt.get_square_asp(ax[i]))

fig.tight_layout()

plt.show()