"""
Cache stats over course of session for sites that meet strict exclusion criteria
(standard, false alarm corr., and DI btwn targets). Also, save the number of 
presentations of each target in each window as a way to screen for windows that
have "nice" balanced data. Post-hoc, use DI/HR/nMisses to screen for valid windows.
e.g. don't want to keep a late window where there were 10 targets but 8 misses. Probabaly
means the animal zoned out at the end.
"""

import os
import pandas as pd
import nems.db as nd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from nems_lbhb.baphy_experiment import BAPHYExperiment
import scipy.ndimage.filters as sf


# ======================= KEEP CRITERIA =============================
# CRD parms
N = 20               # number of target trials that are required of each target class
window_size = 20     # force the window size to be large for this. want reliable di/li measurements
min_good_trials = 35 # minumum number of total "good" baphy trials

# DRX parms
N = 15               # number of target trials that are required of each target class
window_size = 20     # force the window size to be large for this. want reliable di/li measurements
min_good_trials = 35 # minumum number of total "good" baphy trials
# ======================================================================

fpath = '/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/figures/'
dfpath = '/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/results/'
runclass = 'BVT'
animal = 'Cordyceps'
earliest_date = '2019_11_09'

animal = 'Drechsler'
earliest_date = '2019_10_14'

an_regex = "%"+animal+"%"
min_trials = 50

# get list of all training parmfiles
sql = "SELECT parmfile, resppath, gDataRaw.id, cellid, gData.svalue, gData.value FROM gDataRaw INNER JOIN gData on (gDataRaw.id=gData.rawid and gData.name='Tar_Frequencies') WHERE runclass=%s and resppath like %s and training = 1 and bad=0 and trials>%s and behavior='active'"
parmfiles1 = nd.pd_query(sql, (runclass, an_regex, min_trials))
parmfiles1 = parmfiles1.set_index('id')

sql = "SELECT parmfile, resppath, gDataRaw.id, cellid, gData.svalue FROM gDataRaw INNER JOIN gData on (gDataRaw.id=gData.rawid and gData.name='Behave_PumpDuration') WHERE runclass=%s and resppath like %s and training = 1 and bad = 0 and trials>%s and behavior='active'"
parmfiles = nd.pd_query(sql, (runclass, an_regex, min_trials))
parmfiles = parmfiles.set_index('id')
parmfiles = parmfiles.rename(columns={'svalue': 'pumpdur'})

parmfiles = pd.concat([parmfiles['pumpdur'], parmfiles1], axis=1, join='outer')

# screen for dates
parmfiles['date'] = [dt.datetime.strptime('-'.join(x.split('_')[1:-2]), '%Y-%m-%d') for x in parmfiles.parmfile]
ed = dt.datetime.strptime(earliest_date, '%Y_%m_%d')
parmfiles = parmfiles[parmfiles.date > ed]

options = {'pupil': False, 'rasterfs': 100}

# unique id for each session based on the target frequenicies and reward values
parmfiles.loc[parmfiles.svalue.isnull(),'svalue']=parmfiles.loc[parmfiles.svalue.isnull(),'value'].astype(str)
parmfiles['svalue'] = parmfiles['svalue'].str.strip('[]')
parmfiles['pumpdur'] = parmfiles['pumpdur'].str.strip('[]')
parmfiles['svalue']=parmfiles['svalue'].str.replace('\.0','')
parmfiles['svalue']=parmfiles['svalue'].str.replace(' ', '+')
parmfiles['pumpdur']=parmfiles['pumpdur'].str.replace('\.0','')
parmfiles['pumpdur']=parmfiles['pumpdur'].str.replace(' ', '+')

parmfiles['session']=parmfiles.date.dt.strftime("%Y-%m-%d")+"_"+parmfiles['svalue']+"_"+parmfiles['pumpdur']

df = pd.DataFrame()

# now, loop over each unique session and decide if it qualifies to keep based on 
# parameters set above
for session in parmfiles.session.unique():
    files = [p for i, p in parmfiles.iterrows() if p.session==session]
    date = files[0].date
    files = [f['resppath']+f['parmfile'] for f in files]

    manager = BAPHYExperiment(files)
    rec = manager.get_recording(**options)
    rec = rec.and_mask('INVALID_BAPHY_TRIAL', invert=True)

    # search for the number of good trials in which a target was actually presented (i.e. not a false alarm)
    PumpDur = np.array([float(i) for i in session.split('_')[-1].split('+')])
    TargetFreq = np.array([int(i) for i in session.split('_')[-2].split('+')])
    targets = [t for t in rec.epochs.name.unique() if 'TAR_' in t]
    tar_folded = rec['fileidx'].extract_epochs(targets, mask=rec['mask'])
    rew_tars = TargetFreq[PumpDur>0]
    nr_tars = TargetFreq[PumpDur==0]
    rew_tar_str = ['TAR_'+str(t) for t in rew_tars]
    rew_tar_str = [t for t in rew_tar_str if t in tar_folded.keys()]
    nr_tar_str = ['TAR_'+str(t) for t in nr_tars]
    nr_tar_str = [t for t in nr_tar_str if t in tar_folded.keys()]
    
    # determine sufficient targets
    rew_tar_count = 0
    nr_tar_count = 0
    for t in rew_tar_str+nr_tar_str:
        if t in rew_tar_str:
            rew_tar_count += tar_folded[t].shape[0]
        elif t in nr_tar_str:
            nr_tar_count += tar_folded[t].shape[0]
        else:
            raise ValueError("?")
    
    skip = False    # keep this session until we find it doesn't meet criteria
    if (rew_tar_count < N) | (nr_tar_count < N):
        skip = True

    print("NUMBER OF NR TARS: {}".format(nr_tar_count))
    print("NUMBER OF R TARS: {}".format(rew_tar_count))

    # find valid trial numbers using rec epochs times
    invalid_time = rec['fileidx'].get_epoch_bounds('INVALID_BAPHY_TRIAL')
    trial_epochs = rec['fileidx'].get_epoch_bounds('TRIAL')
    rew_tar_times = rec['fileidx'].get_epoch_bounds(rew_tar_str)
    nr_tar_times = rec['fileidx'].get_epoch_bounds(nr_tar_str)
    good_trials = [i+1 for i in range(0, trial_epochs.shape[0]) if (trial_epochs[i] not in invalid_time)]
    print("NUMBER OF GOOD TRIALS: {}".format(len(good_trials)))
    if len(good_trials) < min_good_trials:
        skip = True
    
    # get actual target trial numbers
    rew_trials = []
    nr_trials = []
    for trial in range(0, trial_epochs.shape[0]):
        for rew_time in rew_tar_times:
            if (rew_time[0]>=trial_epochs[trial][0]) & (rew_time[1]<=trial_epochs[trial][1]):
                if (trial+1) in good_trials:
                    rew_trials.append(trial+1)
        
        for nr_time in nr_tar_times:
            if (nr_time[0]>=trial_epochs[trial][0]) & (nr_time[1]<=trial_epochs[trial][1]):
                if (trial+1) in good_trials:
                    nr_trials.append(trial+1)

    # concatenate all target trials
    target_trials = np.sort(nr_trials + rew_trials)

    # if skip is False, go ahead and calculate metrics
    if not skip:
        trialidx = np.arange(0, len(good_trials) - window_size)
        for idx in trialidx:
            print("{0} / {1} windows".format(idx+1, len(trialidx)))
            trials = target_trials[idx:(idx+window_size)]

            # count nRew  / nNRew targets in this window
            n_NR = np.sum([True for t in trials if t in nr_trials])
            n_R = np.sum([True for t in trials if t in rew_trials])

            # note that a hit can also be a FAR. So if n_NR + n_R = window_size and
            # FAR != 0, this is why
            out = manager.get_behavior_performance(trials=trials, **options)

            RTargetStr = [t.strip('TAR_') for t in rew_tar_str]
            NRTargetStr = [t.strip('TAR_') for t in nr_tar_str]

            # calculate LRI for early trials
            HR = np.nansum([out['RR'][t]*out['nTrials'][t] for t in RTargetStr]) / \
                np.nansum([out['nTrials'][t] for t in RTargetStr])
            try:
                if np.nansum([out['nTrials'][t] for t in NRTargetStr]) > 0:
                    HR2 = np.nansum([out['RR'][t]*out['nTrials'][t] for t in NRTargetStr]) / \
                        np.nansum([out['nTrials'][t] for t in NRTargetStr])
            except:
                HR2 = 0
            FAR = out['RR']['Reference']

            # cache LIs in df        
            LI = (HR-HR2) / (HR+HR2)
            if np.isinf(LI):
                LI = 0
            LI_far = ((HR - FAR)- (HR2 - FAR)) / ((HR - FAR) + (HR2 - FAR))
            if np.isinf(LI):
                LI_far = 0

            k = [k for k in out['LI'].keys() if (k.split('_')[0] == rew_tar_str[0][4:]) & (k.split('_')[1] == nr_tar_str[0][4:])][0]
        
            _df = pd.DataFrame({'session': session, 'date': date, 'RTar': rew_tar_str[0], 'NRTar': nr_tar_str[0],
                        'HR': HR, 'HR2': HR2, 'FAR': FAR, 'n_R': n_R, 'n_NR': n_NR,
                        'LI': LI, 'LI_far': LI_far, 'LI_di': out['LI'][k]}, index=[session])

            df = df.append(_df, ignore_index=True)
    
    else:
        print('skipping session: {}'.format(session))


df.to_pickle(dfpath + animal + '_strictSlidingWindow.pickle')