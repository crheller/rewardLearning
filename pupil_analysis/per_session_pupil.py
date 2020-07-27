"""
For a single behavior session plot overall pupil trace, 
and target evoked pupil (separated out by behavioral response).

Normalize to baseline for one plot, plot raw pupil sizes in other.

Include passive data
""" 

from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems.db as nd
import datetime as dt
import numpy as np
import charlieTools.reward_learning.behavior_helpers as bhelp

import matplotlib.pyplot as plt

date = "2020_07_07"
runclass = 'TBP'
animal = 'Cordyceps'
rasterfs = 100
options = {'rasterfs': rasterfs, 'pupil': True}

# get list of all training parmfiles
parmfiles = bhelp.get_training_files(animal, runclass, date)

fns = [r+p for r, p in zip(parmfiles['resppath'], parmfiles['parmfile'])]

manager = BAPHYExperiment(fns)
rec = manager.get_recording(**options)


# plot pupil results

f = plt.figure(figsize=(12, 6))

ptax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
rtax = plt.subplot2grid((2, 2), (1, 0))
ntax = plt.subplot2grid((2, 2), (1, 1))

# plot continuous trace
ptax.plot(rec['pupil']._data.T)

# plot raw trial averaged
behave_outcomes = ['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL', 'PASSIVE_EXPERIMENT']
colors = ['red', 'blue', 'blue', 'red', 'k']
linestyles = ['-', '-', '--', '--', '-']
for bh, c, ls in zip(behave_outcomes, colors, linestyles):
    r = rec.copy()
    r = r.and_mask([bh])
    d = r['pupil'].extract_epoch('TARGET', mask=r['mask'], allow_incomplete=True)

    m = np.nanmean(d, axis=(0, 1))
    sem = np.nanstd(d, axis=(0, 1)) / np.sqrt(d.shape[0])
    t = np.linspace(0, d.shape[-1] / rasterfs, d.shape[-1])
    rtax.plot(t, m, linestyle=ls, color=c, label=bh)
    rtax.fill_between(t, m-sem, m+sem, color=c, alpha=0.3, lw=0)

    # subtract prestim mean on each trial
    r = r.and_mask(['TARGET'])
    dpre = r['pupil'].extract_epoch('PreStimSilence', mask=r['mask'], allow_incomplete=True)
    dpre = np.nanmean(dpre, axis=(1, -1), keepdims=True)
    d = d - dpre

    m = np.nanmean(d, axis=(0, 1))
    sem = np.nanstd(d, axis=(0, 1)) / np.sqrt(d.shape[0])
    t = np.linspace(0, d.shape[-1] / rasterfs, d.shape[-1])
    ntax.plot(t, m, linestyle=ls, color=c, label=bh)
    ntax.fill_between(t, m-sem, m+sem, color=c, alpha=0.3, lw=0)


ntax.set_xlabel('Time (s)')
rtax.set_xlabel('Time (s)')
ntax.set_ylabel('Pupil size \n (normalized to Target PreStim)')
rtax.set_ylabel('Pupil size (raw)')
rtax.legend(frameon=False)

f.tight_layout()

# plot per trial pupil heatmaps (normalized to trial start and raw), sorted by trial length
behavior_outcomes = ['HIT_TRIAL', 'INCORRECT_HIT_TRIAL', 'FALSE_ALARM_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL']
ymax = 75
nmin = -40
nmax = 40
f2, ax2 = plt.subplots(1, 5, figsize=(12, 8))
for b, a2 in zip(behavior_outcomes, ax2):
    p = rec['pupil'].extract_epoch(b)
    tlen = [sum(~np.isnan(p[i,0,:])) for i in range(p.shape[0])]  
    idx = np.argsort(tlen)
    p = p[idx, 0, :]

    # normalize to trial onset
    m = np.nanmean(p[:, :20], axis=-1)
    p = (p.T - m).T
    a2.imshow(p, cmap='PiYG', aspect='auto', vmin=nmin, vmax=nmax)
    a2.set_title(b)
    a2.set_xlim((0, 1000))
    a2.set_ylim((0, ymax))

f2.tight_layout()

# plot per trial heatmaps for passive, sorted by target category
targets = [e for e in rec.epochs.name.unique() if 'TAR_' in e]
ymax = 100
nmin = -30
nmax = 30

for t in targets:
    if t not in rec.signals.keys():
        rec[t] = rec['pupil'].epoch_to_signal(t)

f, ax = plt.subplots(1, len(targets), figsize=(8, 6))
if len(targets) > 1:
    pass
else: ax = [ax]

for a, t in zip(ax, targets):

    r = rec.copy()
    r = r.and_mask(['PASSIVE_EXPERIMENT'])
    p = r['pupil'].extract_epoch('TRIAL', mask=r['mask'])
    tar_mask = r[t].extract_epoch('TRIAL', mask=r['mask'])
    mask = tar_mask[:, 0, :].sum(axis=(-1)) > 0
    p = p[mask, :, :]
    tlen = [sum(~np.isnan(p[i,0,:])) for i in range(p.shape[0])]  
    idx = np.argsort(tlen)
    p = p[idx, 0, :]

    # normalize to trial onset
    m = np.nanmean(p[:, :20], axis=-1)
    p = (p.T - m).T
    a.imshow(p, cmap='PiYG', aspect='auto', vmin=nmin, vmax=nmax)
    a.set_title('PASSIVE TRIAL, {}'.format(t))
    a.set_xlim((0, 700))
    a.set_ylim((0, ymax))

f.tight_layout()


# get lick triggered pupil
f, ax = plt.subplots(1, 2, figsize=(12, 5))
te = 5  # seconds
teidx = te * rasterfs
ts = 2  # seconds prior to lick
tsidx = ts * rasterfs
licks = rec['pupil'].get_epoch_indices('LICK')[:, 0]
ttypes = ['HIT_TRIAL', 'INCORRECT_HIT_TRIAL', 'FALSE_ALARM_TRIAL', 'PASSIVE_EXPERIMENT'] # trials with licks
for ttype in ttypes:
    r = rec.copy()
    r = r.and_mask(ttype)
    trial_indices = r['pupil'].get_epoch_indices('TRIAL', mask=r['mask'])
    t_licks = np.array([licks[np.argwhere((licks>=ti[0]) & (licks<=ti[1]))][0][0] for ti in trial_indices
                             if np.argwhere((licks>=ti[0]) & (licks<=ti[1])).size > 0])

    indices = np.stack([t_licks-tsidx, t_licks+teidx]).T
    indices = indices[indices[:,0]>0]
    data = np.zeros((indices.shape[0], tsidx+teidx))
    for i, (lb, ub) in enumerate(indices):
        data[i, :] = rec['pupil']._data[0, lb:ub]


    t = np.linspace(-ts, te, data.shape[-1])
    m = data.mean(axis=0)
    sem = data.std(axis=0) / np.sqrt(data.shape[0])
    ax[0].plot(t, m, label=ttype)
    ax[0].fill_between(t, m-sem, m+sem, alpha=0.3, lw=0)

    # normalize to 2 s pre-lick
    data = (data.T - data[:, :20].mean(axis=-1)).T
    t = np.linspace(-ts, te, data.shape[-1])
    m = data.mean(axis=0)
    sem = data.std(axis=0) / np.sqrt(data.shape[0])
    ax[1].plot(t, m, label=ttype)
    ax[1].fill_between(t, m-sem, m+sem, alpha=0.3, lw=0)


ax[0].axvline(0, linestyle='--', color='k', label='lick onset')
ax[0].legend(frameon=False)
ax[0].set_xlabel('Time (s)')   
ax[0].set_ylabel('Pupil size') 

ax[1].axvline(0, linestyle='--', color='k', label='lick onset')
ax[1].legend(frameon=False)
ax[1].set_xlabel('Time (s)')   
ax[1].set_ylabel('Pupil size (normalized)') 

f.tight_layout()

plt.show()