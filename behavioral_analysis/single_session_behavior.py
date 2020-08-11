# Example behavior analyses for rewardTargetLBHB behaviors:
#   two tone discrimnation / variable reward
#   variable SNR pure tone detect
#   single pure tone detect
# CRH - 10/06/2019


import nems_lbhb.behavior as beh
import matplotlib.pyplot as plt
import numpy as np
import nems_lbhb.baphy_io as io
from nems_lbhb.baphy_experiment import BAPHYExperiment
import os

def get_square_asp(ax):
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    return asp

options = {'pupil': False, 'rasterfs': 100}
window_length = 20
cum = False

p1 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_03_10_BVT_5.m'
p2 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_03_10_BVT_8.m'
p3 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_03_09_BVT_1.m'
p4 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_03_06_BVT_1.m'

p1 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_07_07_TBP_1.m'
p2 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_07_07_TBP_3.m'
p3 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_07_06_TBP_5.m'

# with catch trials (need to do something about the "cue" target -- 
# shouldn't be lumped with the real target, probabaly)
p1 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_07_24_TBP_1.m'
p2 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_07_24_TBP_3.m'
p3 =  '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_07_24_TBP_5.m'

#p3 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_06_26_BVT_5.m'
#p4 = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_06_25_BVT_7.m'

parmfiles = [[p1, p2]]

for pf in parmfiles:
    manager = BAPHYExperiment(pf)
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
    if type(pf) is list:
        file_epochs = rec['fileidx'].get_epoch_bounds([e for e in rec.epochs.name.unique() if 'FILE_' in e])
        file_dividers = [np.argwhere(np.diff(trial_epochs[:, 1]<=f[1]))[0][0] for f in file_epochs[:-1]]

    step = 2
    if cum:
        trialidx = np.arange(0, len(good_trials), step)
    else:
        trialidx = np.arange(0, len(good_trials)-window_length+step, step)
    trialcount = len(trialidx)
    HR = np.zeros(trialcount)
    HR2 = np.zeros(trialcount)
    FAR = np.zeros(trialcount)
    LI = np.zeros(trialcount)
    LI_di = np.zeros(trialcount)
    LI_far = np.zeros(trialcount)

    # find rewarded target using the unique session id
    params = manager.get_baphy_exptparams()[0]
    PumpDur = np.array(params['BehaveObject'][1]['PumpDuration'])
    TargetFreq = np.array(params['TrialObject'][1]['TargetHandle'][1]['Names'])
    RTarget = TargetFreq[PumpDur>0]
    RTargetStr = [str(r) for r in RTarget]
    NRTarget = TargetFreq[PumpDur==0]
    NRTargetStr = [str(r) for r in NRTarget]

    for i, s in enumerate(trialidx):
        print("{0} / {1}".format(s, len(good_trials)))
        e = np.min([s+window_length, len(good_trials)])
        if cum:
            start = 0
        else:
            start = s
        trials = good_trials[start:e]
        out = manager.get_behavior_performance(trials=trials, **options)
        try:
            k = [k for k in out['LI'].keys() if ','.join(RTargetStr)==k.split('_')[0]][0]
        except:
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


    f, ax = plt.subplots(1, 1, figsize=(8, 3))

    tt = np.array(good_trials)[trialidx]
    ax.plot(tt, HR, '.-', label=targets[rtidx])
    ax.plot(tt, HR2, '.-', label=targets[nrtidx])
    ax.plot(tt, FAR, '.-', label='FAR')
    ax.plot(tt, LI, '--', label='LI')
    #ax.plot(tt, LI_far, '--', label='LI_far')
    ax.plot(tt, LI_di, '--', label='LI_di')
    ax.axhline(0.5, lw=2, color='red')

    ax.set_ylim((0, 1.1))
    ax.set_ylabel('Hit rate')
    ax.legend(frameon=False)
    if type(pf) == list:
        ax.set_title(os.path.basename(pf[0]))
    else:
        ax.set_title(os.path.basename(pf))

    if type(pf) is list:
        for d in file_dividers:
            ax.axvline(tt.flat[np.abs(tt - d).argmin() - int(window_length / step)] , color='k', lw=2)

    f.tight_layout()

plt.show()
