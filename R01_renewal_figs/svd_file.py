"""
Load all Cordy BVT files with pupil (gDataRaw.eyewin=2). Calculate LI over sliding window. Plot LI vs. pupil size.
Normalize pupil size per day.
"""
import os
import pandas as pd
import nems.db as nd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from nems_lbhb.baphy_experiment import BAPHYExperiment
import scipy.ndimage.filters as sf

fpath = '/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/figures/'
dfpath = '/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/results/'
#fpath = '/auto/users/svd/projects/reward_training/behavior/'
runclass = 'BVT'
animal = 'Cordyceps'
earliest_date = '2019_11_09'

#animal = 'Drechsler'
#earliest_date = '2019_10_14'

an_regex = "%"+animal+"%"
min_trials = 50

single_file = False
if single_file:
    filename = "DRX007a12"
    datestr = "2019-11-14"
    filename = "DRX006b"
    datestr = "2019-11-12"
    filename = "DRX008b10_a_BVT"
    datestr = "2019-11-15"
    filename = "DRX005c"
    datestr = "2019-11-11"
    filename = "Drechsler_2019_10_28_BVT_1"
    datestr = '2019-10-28'

    # day that gives trouble
    #filename="Cordyceps_2019_11_14"
    #datestr = "2019-11-14"
    filename = 'Cordyceps_2019_11_27_BVT'
    datestr = '2019-11-27'

    sql = "SELECT parmfile, resppath, gDataRaw.id, cellid, gData.svalue, gData.value FROM gDataRaw INNER JOIN gData on (gDataRaw.id=gData.rawid and gData.name='Tar_Frequencies') WHERE runclass=%s and behavior='active' and parmfile like %s"
    parmfiles1 = nd.pd_query(sql, (runclass, filename+"%"))
    parmfiles1['date'] = dt.datetime.strptime(datestr, '%Y-%m-%d')
    parmfiles1 = parmfiles1.set_index('id')
    
    sql = "SELECT parmfile, resppath, gDataRaw.id, cellid, gData.svalue FROM gDataRaw INNER JOIN gData on (gDataRaw.id=gData.rawid and gData.name='Behave_PumpDuration') WHERE runclass=%s and behavior='active' and parmfile like %s"
    parmfiles = nd.pd_query(sql, (runclass, filename+"%"))
    parmfiles['date'] = dt.datetime.strptime(datestr, '%Y-%m-%d')
    parmfiles = parmfiles.set_index('id')
    parmfiles = parmfiles.rename(columns={'svalue': 'pumpdur'})

    # now join on rawid
    parmfiles = pd.concat([parmfiles['pumpdur'], parmfiles1], axis=1, join='outer')

else:
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
window_length = 20

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

#for file in parmfiles.parmfile.unique():
for session in parmfiles.session.unique():
    #files = [p for i, p in parmfiles.iterrows() if p.parmfile==file]
    files = [p for i, p in parmfiles.iterrows() if p.session==session]
    date = files[0].date
    files = [f['resppath']+f['parmfile'] for f in files]
    file = files[0]

    manager = BAPHYExperiment(files)
    rec = manager.get_recording(**options)

    # compute HR's over sliding window in baphy trial time
    # only on valid trials
    totalTrials = rec['fileidx'].extract_epoch('TRIAL').shape[0]
    rec = rec.and_mask('INVALID_BAPHY_TRIAL', invert=True)
    validTrials = rec['fileidx'].extract_epoch('TRIAL', mask=rec['mask']).shape[0]

    # find valid trial numbers using rec epochs times
    invalid_time = rec['fileidx'].get_epoch_bounds('INVALID_BAPHY_TRIAL')
    trial_epochs = rec['fileidx'].get_epoch_bounds('TRIAL')
    good_trials = [i+1 for i in range(0, trial_epochs.shape[0]) if trial_epochs[i] not in invalid_time]

    step = 2
    trialidx = np.arange(0, len(good_trials)-window_length+step, step)
    trialcount = len(trialidx)
    HR = np.zeros(trialcount)
    HR2 = np.zeros(trialcount)
    FAR = np.zeros(trialcount)
    LI = np.zeros(trialcount)
    LI_di = np.zeros(trialcount)
    LI_far = np.zeros(trialcount)

    # find rewarded target using the unique session id
    PumpDur = np.array([float(i) for i in session.split('_')[-1].split('+')])
    TargetFreq = np.array([int(i) for i in session.split('_')[-2].split('+')])
    RTarget = TargetFreq[PumpDur>0]
    RTargetStr = [str(r) for r in RTarget]
    NRTarget = TargetFreq[PumpDur==0]
    NRTargetStr = [str(r) for r in NRTarget]

    for i,s in enumerate(trialidx):
        print("{0} / {1}".format(s, len(good_trials)))
        e = np.min([s+window_length, len(good_trials)])
        trials = good_trials[s:e]
        out = manager.get_behavior_performance(trials=trials, **options)
        k = [k for k in out['LI'].keys() if '+'.join(RTargetStr)==k.split('_')[0]][0]
        LI_di[i] = out['LI'][k]

        # deal with multiple targets
        HR[i] = np.nansum([out['RR'][t]*out['nTrials'][t] for t in RTargetStr]) / \
            np.nansum([out['nTrials'][t] for t in RTargetStr])
        try:
            if np.nansum([out['nTrials'][t] for t in NRTargetStr]) > 0:
                HR2[i] = np.nansum([out['RR'][t]*out['nTrials'][t] for t in NRTargetStr]) / \
                    np.nansum([out['nTrials'][t] for t in NRTargetStr])
        except:
            HR2[i] = 0
        FAR[i] = out['RR']['Reference']

    LI = (HR-HR2) / (HR+HR2)
    LI[np.isinf(LI)] = 0
    LI_far = ((HR - FAR)- (HR2 - FAR)) / ((HR - FAR) + (HR2 - FAR))
    LI_far[np.isinf(LI_far)] = 0

    targets = [t for t in out['RR'].keys() if 'Reference' not in t]
    for i, t in enumerate(targets):
        if t in RTargetStr:
            targets[i] = targets[i]+"*"
            rtidx = i
        else:
            nrtidx = i

    session = "_".join([str(date.date()), RTargetStr[0], NRTargetStr[0]])

    _df = pd.DataFrame({'session': session, 'date': date, 'RTar': RTarget[0], 'NRTar': NRTarget[0],
                        'HR': HR, 'HR2': HR2, 'FAR': FAR, 'LI': LI, 'LI_di': LI_di, 'LI_far': LI_far})
    df = df.append(_df, ignore_index=True)

    f, ax = plt.subplots(2, 1)

    tt = np.array(good_trials)[trialidx]
    ax[0].plot(tt, HR, '.-', label=targets[rtidx])
    ax[0].plot(tt, HR2, '.-', label=targets[nrtidx])
    ax[0].plot(tt, FAR, '.-', label='FAR')
    ax[0].plot(tt, LI, '--', label='LI')
    ax[0].plot(tt, LI_far, '--', label='LI_far')
    ax[0].plot(tt, LI_di, '--', label='LI_di')

    ax[0].set_ylim((0, 1.1))
    ax[0].set_ylabel('Hit rate')
    #ax[0].set_xlabel('Baphy Trials')
    ax[0].legend()
    ax[0].set_title(os.path.basename(parmfiles[parmfiles.date==date]['parmfile'].values[0]))
    ax[0].set_title(os.path.basename(file))

    if not single_file:
        f.savefig(fpath + os.path.basename(file) + '.png')
        plt.close(f)


# stash full dataframe
if not single_file:
    df.to_pickle(dfpath+'{}_slidingWindow.pickle'.format(animal))