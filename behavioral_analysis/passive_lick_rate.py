"""
Inspired by Kuchibholta et al., 2019, see if show evidence of 
reward learning during passive blocks. To do this, 
simply plot the lick rate for each target (and references)
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sf

mfile = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_06_18_BVT_3.m'
mfile = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_07_02_TBP_4.m'

manager = BAPHYExperiment(mfile)
rec = manager.get_recording(**{'rasterfs': 100})

lick_signal = rec['fileidx'].epoch_to_signal('LICK', onsets_only=True)

tars = [e for e in rec.epochs.name.unique() if 'TAR_' in e]
tars = [str(t) for t in np.sort(tars)]
tar1_lick = lick_signal.extract_epoch(tars[0])
tar2_lick = lick_signal.extract_epoch(tars[1])
ref_lick =  lick_signal.extract_epoch('REFERENCE')

f, ax = plt.subplots(1, 3, figsize=(5, 5))

ax[0].imshow(tar1_lick.squeeze(), aspect='auto')
ax[0].set_title(tars[0])

ax[1].imshow(tar2_lick.squeeze(), aspect='auto')
ax[1].set_title(tars[1])

ax[2].imshow(ref_lick.squeeze(), aspect='auto')
ax[2].set_title('REFERENCE')

f.tight_layout()


sigma = 1.2
f, ax = plt.subplots(1, 1, figsize=(5, 3))

tart = np.linspace(0, tar1_lick.shape[-1]/rec.meta['rasterfs'], tar1_lick.shape[-1])
reft = np.linspace(0, ref_lick.shape[-1]/rec.meta['rasterfs'], ref_lick.shape[-1])
sound_onset = rec['fileidx'].extract_epoch('PreStimSilence').shape[-1] / rec.meta['rasterfs']
sound_offset = (ref_lick.shape[-1] - rec['fileidx'].extract_epoch('PreStimSilence').shape[-1]) / rec.meta['rasterfs']

tar1mean = sf.gaussian_filter1d(np.nanmean(tar1_lick, axis=(0, 1)), sigma=sigma, axis=-1)
tar2mean = sf.gaussian_filter1d(np.nanmean(tar2_lick, axis=(0, 1)), sigma=sigma, axis=-1)
refmean = sf.gaussian_filter1d(np.nanmean(ref_lick, axis=(0, 1)), sigma=sigma, axis=-1)

ax.plot(tart, tar1mean, label=tars[0])
ax.plot(tart, tar2mean, label=tars[1])
ax.plot(reft, refmean, label='REF')
ax.axvline(sound_onset, linestyle='--', color='k', lw=2)
ax.axvline(sound_offset, linestyle='--', color='k', lw=2)

ax.set_ylabel('Lick rate')
ax.set_xlabel('Time')
ax.legend(frameon=False)
f.tight_layout()

plt.show()

