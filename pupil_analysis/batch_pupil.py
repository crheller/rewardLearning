"""
Load pupil for all training sessions. 
Two analyses:
    1) within trials (e.g. normalize per trial)
        align to trial onset
        align to lick onset
        align to target onset
    2) Over blocks on given day (show DI as function of pupil size?)
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems.db as nd
import datetime as dt
import numpy as np
import charlieTools.reward_learning.behavior_helpers as bhelp

import matplotlib.pyplot as plt

early_date = "2020_06_15"
late_date = "2020_07_24"
runclass = 'TBP'
animal = 'Cordyceps'
rasterfs = 20
options = {'rasterfs': rasterfs, 'pupil': True}

pre_event_time =  1 # seconds 
post_event_time = 6 # seconds
preidx = int(pre_event_time * rasterfs)
postidx = int(post_event_time * rasterfs)

# get list of all training parmfiles
parmfiles = bhelp.get_training_files(animal, runclass, 
                                      earliest_date=early_date, latest_date=late_date, pupil=True,
                                      min_trials=40)

fns = [r+p for r, p in zip(parmfiles['resppath'], parmfiles['parmfile'])]
udate = np.unique([s.split(animal+'_')[1].split('_'+runclass)[0] for s in fns])

# combine by date
expts = []
for d in udate:
    x = []
    for f in fns:
        if d in f:
            x.append(f)
    expts.append(x)

lick_triggered = {
    'HIT_TRIAL': [],
    'INCORRECT_HIT_TRIAL': [],
    'FALSE_ALARM_TRIAL': [],
    'PASSIVE_EXPERIMENT': []
}
trial_triggered = {
    'HIT_TRIAL': [],
    'INCORRECT_HIT_TRIAL': [],
    'FALSE_ALARM_TRIAL': [],
    'PASSIVE_EXPERIMENT': [],
    'MISS_TRIAL': [],
    'CORRECT_REJECT_TRIAL': []
}
sound_triggered = {
    'HIT_TRIAL': [],
    'INCORRECT_HIT_TRIAL': [],
    'FALSE_ALARM_TRIAL': [],
    'PASSIVE_EXPERIMENT': [],
    'MISS_TRIAL': [],
    'CORRECT_REJECT_TRIAL': []
}
for expt in expts:
    # for each day, load the training recording with pupil, compute
    # lick triggered pupil, target onset triggered pupil, trial onset triggered pupil
    # each split by trial outcome (False Alarm, Hit etc.)
    manager = BAPHYExperiment(expt)
    rec = manager.get_recording(**options)


    # lick triggered
    licks = rec['pupil'].get_epoch_indices('LICK')[:, 0]
    ttypes = lick_triggered.keys()
    for ttype in ttypes:
        r = rec.copy()
        r = r.and_mask(ttype)
        # figure out which trials have licks, then get start / stop indices based on pre/post event times
        trial_indices = r['pupil'].get_epoch_indices('TRIAL', mask=r['mask'])
        t_licks = np.array([licks[np.argwhere((licks>=ti[0]) & (licks<=ti[1]))][0][0] for ti in trial_indices
                                if np.argwhere((licks>=ti[0]) & (licks<=ti[1])).size > 0])

        indices = np.stack([t_licks-preidx, t_licks+postidx]).T
        indices = indices[indices[:,0] > 0]
        data = np.zeros((indices.shape[0], preidx+postidx))
        for i, (lb, ub) in enumerate(indices):
            data[i, :] = rec['pupil']._data[0, lb:ub]
        
        # normalize to pre event period
        data = (data.T - data[:, :preidx].mean(axis=-1)).T
        m = data.mean(axis=0)
        lick_triggered[ttype].append(m)


    # trial triggered
    ttypes = trial_triggered.keys()
    for ttype in ttypes:
        r = rec.copy()
        r = r.and_mask(ttype)
        trial_indices = r['pupil'].get_epoch_indices('TRIAL', mask=r['mask'])[:, 0]
        indices = np.stack([trial_indices-preidx, trial_indices+postidx]).T
        indices = indices[indices[:,0] > 0]
        data = np.zeros((indices.shape[0], preidx+postidx))
        for i, (lb, ub) in enumerate(indices):
            try:
                data[i, :] = rec['pupil']._data[0, lb:ub]
            except ValueError:
                pad = data.shape[-1] - rec['pupil']._data[0, lb:ub].shape[0]
                pad = np.nan * np.ones(pad)
                vals = np.concatenate((rec['pupil']._data[0, lb:ub], pad))
                data[i, :] = vals
        
        # normalize to pre event period
        data = (data.T - data[:, :preidx].mean(axis=-1)).T
        m = data.mean(axis=0)
        trial_triggered[ttype].append(m)


    # TODO sound triggered
    ttypes = sound_triggered.keys()
    for ttype in ttypes:
        r = rec.copy()
        r = r.and_mask(ttype)
        trial_indices = r['pupil'].get_epoch_indices('TRIAL', mask=r['mask'])[:, 0]
        indices = np.stack([trial_indices-preidx, trial_indices+postidx]).T
        indices = indices[indices[:,0] > 0]
        data = np.zeros((indices.shape[0], preidx+postidx))
        for i, (lb, ub) in enumerate(indices):
            try:
                data[i, :] = rec['pupil']._data[0, lb:ub]
            except ValueError:
                pad = data.shape[-1] - rec['pupil']._data[0, lb:ub].shape[0]
                pad = np.nan * np.ones(pad)
                vals = np.concatenate((rec['pupil']._data[0, lb:ub], pad))
                data[i, :] = vals
        
        # normalize to pre event period
        data = (data.T - data[:, :preidx].mean(axis=-1)).T
        m = data.mean(axis=0)
        sound_triggered[ttype].append(m)



t = np.linspace(-pre_event_time, post_event_time, preidx+postidx)
f, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

# plot lick triggered pupil
for k in lick_triggered.keys():
    m = np.nanmean(np.stack(lick_triggered[k]), 0)
    sem = np.nanstd(np.stack(lick_triggered[k]), 0) / np.sqrt(len(lick_triggered[k]))

    ax[0].plot(t, m, label=k)
    ax[0].fill_between(t, m-sem, m+sem, alpha=0.3, lw=0)

ax[0].set_xlabel('Time from lick onset')
ax[0].set_ylabel('Normalized (to pre-lick) pupil')
ax[0].axvline(0, linestyle='--', color='k')
ax[0].set_title("LICK-TRIGGERED Pupil Size")


# plot trial triggered pupil
for k in trial_triggered.keys():
    m = np.nanmean(np.stack(trial_triggered[k]), 0)
    sem = np.nanstd(np.stack(trial_triggered[k]), 0) / np.sqrt(len(trial_triggered[k]))

    ax[1].plot(t, m, label=k)
    ax[1].fill_between(t, m-sem, m+sem, alpha=0.3, lw=0)

ax[1].set_xlabel('Time from trial onset')
ax[1].set_ylabel('Normalized (to pre-trial) pupil')
ax[1].axvline(0, linestyle='--', color='k')
ax[1].set_title("TRIAL-TRIGGERED Pupil Size")


# plot sound triggered pupil
for k in trial_triggered.keys():
    m = np.nanmean(np.stack(sound_triggered[k]), 0)
    sem = np.nanstd(np.stack(sound_triggered[k]), 0) / np.sqrt(len(sound_triggered[k]))

    ax[2].plot(t, m, label=k)
    ax[2].fill_between(t, m-sem, m+sem, alpha=0.3, lw=0)

ax[2].legend(frameon=False, fontsize=8)
ax[2].set_xlabel('Time from sound onset')
ax[2].set_ylabel('Normalized (to pre-sound) pupil')
ax[2].axvline(0, linestyle='--', color='k')
ax[2].set_title("SOUND-TRIGGERED Pupil Size")

f.tight_layout()

plt.show()