import nems.db as nd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nems_lbhb.baphy_experiment import BAPHYExperiment
import os

runclass = 'BVT'
animal = 'Cordyceps'
an_regex = "%"+animal+"%"
min_trials = 50
earliest_date = '2019_10_03'  # date when we started repeating unrewarded trials
earliest_date = '2019_11_09'
#earliest_date = '2020_06_08'
valid = True
verbose = True
# get list of all training parmfiles
sql = "SELECT parmfile, resppath FROM gDataRaw WHERE runclass=%s and resppath like %s and training = 1 and bad=0 and trials>%s and behavior='active'"
parmfiles = nd.pd_query(sql, (runclass, an_regex, min_trials))

# screen for dates
parmfiles['date'] = [dt.datetime.strptime('-'.join(x.split('_')[1:-2]), '%Y-%m-%d') for x in parmfiles.parmfile]
ed = dt.datetime.strptime(earliest_date, '%Y_%m_%d')
parmfiles = parmfiles[parmfiles.date >= ed]
files = [os.path.join(p[1]['resppath'], p[1]['parmfile']) for p in parmfiles.iterrows()]


cfs = np.round(np.logspace(np.log2(500), np.log2(16000), num=15, base=2), 0)
# store HR, DI, near each cf regardless of reward value.
HR = np.nan * np.zeros((len(files), len(cfs)))
DI = np.nan * np.zeros((len(files), len(cfs)))
# store LI as function of both cfs (rew/n.r.)
LI = np.nan * np.zeros((len(files), len(cfs), len(cfs)))
# store experiment length
exptlen = np.nan * np.zeros(len(files))
# store total repetitions
ref_reps = np.nan * np.zeros(len(files))
tar_reps = np.nan * np.zeros((2, len(files)))
# store trial counts
totalBaphyTrials = np.nan * np.zeros(len(files))
totalValidTrials = np.nan * np.zeros(len(files))

for i, f in enumerate(files):
    manager = BAPHYExperiment(f)
    options = {'pupil': False, 'rasterfs': 100}
    rec = manager.get_recording(**options)

    try:
        out_valid = manager.get_behavior_performance(trials=None, **options)
        
        #options.update({'keep_following_incorrect_trial': True, 'keep_cue_trials': True, 'keep_early_trials': True})
        #out_all = manager.get_behavior_performance(trials=None, **options)

        # get target freqs
        tars = list(out_valid['DI'].keys())
        tars = np.sort(tars)
        if len(tars) == 2:
            t1 = int(tars[0])
            t2 = int(tars[1])

            # get closest bin for each
            idx1 = np.abs(t1 - cfs).argmin()
            idx2 = np.abs(t2 - cfs).argmin()
            tar1 = cfs.flat[idx1] 
            tar2 = cfs.flat[idx2] 

            # HR for each target
            HR[i, idx1] = out_valid['RR'][tars[0]]
            HR[i, idx2] = out_valid['RR'][tars[1]]

            # DI for each target
            DI[i, idx1] = out_valid['DI'][tars[0]]
            DI[i, idx2] = out_valid['DI'][tars[1]]

            # LI (rew. pref)
            exptparams = manager.get_baphy_exptparams()[0]
            pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])
            rew_idx = [True if i>0 else False for i in pump_dur]
            rew_tar = np.array(tars)[rew_idx][0]
            nr_tar = np.array(tars)[~np.array(rew_idx)][0]
            if np.argwhere(rew_idx)[0][0]==1:
                LI[i, idx1, idx2] = (out_valid['RR'][rew_tar] - out_valid['RR'][nr_tar]) / (out_valid['RR'][rew_tar] + out_valid['RR'][nr_tar])
            else:
                LI[i, idx2, idx1] = (out_valid['RR'][rew_tar] - out_valid['RR'][nr_tar]) / (out_valid['RR'][rew_tar] + out_valid['RR'][nr_tar])

            # Trial counts
            totalBaphyTrials[i] = rec.epochs[rec.epochs.name=='TRIAL'].shape[0]
            totalValidTrials[i] = totalBaphyTrials[i] - rec.epochs[rec.epochs.name=='INVALID_BAPHY_TRIAL'].shape[0]

            # Ref counts (unique repetitions)
            r = rec.and_mask(['INVALID_BAPHY_TRIAL'], invert=True)
            min_count = 1e10
            for e in ([e for e in rec.epochs.name.unique() if 'STIM_' in e]):
                try:
                    if r['fileidx'].extract_epoch(e, mask=r['mask']).shape[0] < min_count:
                        min_count = r['fileidx'].extract_epoch(e, mask=r['mask']).shape[0]
                except:
                    min_count = 0

            ref_reps[i] = min_count

            # Tar counts (tar repetitions)
            tar_reps[0, i] = out_valid['nTrials'][rew_tar]
            tar_reps[1, i] = out_valid['nTrials'][nr_tar]

            # experiment length (time, in minutes)
            try:
                minutes = exptparams['StopTime'][4] - exptparams['StartTime'][4]
                if minutes < 0:
                    minutes = 60 - exptparams['StartTime'][4] + exptparams['StopTime'][4]
                exptlen[i] = minutes
            except:
                pass
    
    except ValueError:
        # no behavior in this file?
        pass


f = plt.figure(figsize=(9, 6))
liax = plt.subplot2grid((2, 3), (0, 0))
diax = plt.subplot2grid((2, 3), (0, 1), colspan=2)
trax = plt.subplot2grid((2, 3), (1, 0)) # trials vs time
tarax = plt.subplot2grid((2, 3), (1, 1)) # target reps vs time
refax = plt.subplot2grid((2, 3), (1, 2)) # ref reps vs. time

# plot average learning index as function of rew. and n.r. center freq
liax.imshow(np.nanmean(LI, axis=0), aspect='auto', cmap='PRGn', vmin=-0.5, vmax=0.5, origin='lower')
liax.set_xticks(range(0, len(cfs)))
liax.set_xticklabels(cfs.astype(int), rotation=45, fontsize=8)
liax.set_yticks(range(0, len(cfs)))
liax.set_yticklabels(cfs.astype(int), fontsize=8)
liax.set_title('Learning Index')
liax.set_xlabel('Rewarded CF')
liax.set_ylabel('Non-rewarded CF')

# plot HR/DI (regardless of reward value) vs. center freq
x = range(0, len(cfs))
m = np.nanmean(HR, axis=0)
sem = np.nanstd(HR, axis=0) / np.count_nonzero(~np.isnan(HR), axis=0)
diax.plot(m, 'o-', label='HR')
diax.fill_between(x, m-sem, m+sem, alpha=0.3)
m = np.nanmean(DI, axis=0)
sem = np.nanstd(DI, axis=0) / np.count_nonzero(~np.isnan(DI), axis=0)
diax.plot(m, 'o-', label='DI')
diax.fill_between(x, m-sem, m+sem, alpha=0.3)
diax.set_xticks(x)
diax.set_xticklabels(cfs.astype(int), fontsize=8, rotation=45)
diax.legend(frameon=False)
diax.set_ylabel('HR / DI')

# plot number of trials (valid trials) vs. experiment time
trax.scatter(exptlen, totalValidTrials, s=20, edgecolor='white')
trax.set_xlabel('Time (minutes)')
trax.set_ylabel('Valid Trial Count')

# plot number of ref repetitions vs. experiment time
refax.scatter(exptlen, ref_reps, s=20, edgecolor='white')
refax.set_xlabel('Time (minutes)')
refax.set_ylabel('Minimum REF reps')

# plot number of trials (valid trials) vs. experiment time
tarax.scatter(exptlen, np.nanmean(tar_reps, axis=0), s=20, edgecolor='white')
tarax.set_xlabel('Time (minutes)')
tarax.set_ylabel('Mean TAR reps')

f.canvas.set_window_title('Cordyceps')

f.tight_layout()

plt.show()