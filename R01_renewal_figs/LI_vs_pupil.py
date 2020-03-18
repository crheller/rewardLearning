"""
Load all Cordy BVT files with pupil (gDataRaw.eyewin=2). Calculate LI over sliding window. Plot LI vs. pupil size.
Normalize pupil size per day.
"""
import nems.db as nd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from nems_lbhb.baphy_experiment import BAPHYExperiment

window_length = 15 # in baphy trials

runclass = 'BVT'
animal = 'Cordyceps'
an_regex = "%"+animal+"%"
min_trials = 50
earliest_date = '2019_11_03'

# get list of all training parmfiles
sql = "SELECT parmfile, resppath FROM gDataRaw WHERE runclass=%s and resppath like %s and training = 1 and bad=0 and trials>%s and behavior='active' and eyewin=2"
parmfiles = nd.pd_query(sql, (runclass, an_regex, min_trials))

# screen for dates
parmfiles['date'] = [dt.datetime.strptime('-'.join(x.split('_')[1:-2]), '%Y-%m-%d') for x in parmfiles.parmfile]
ed = dt.datetime.strptime(earliest_date, '%Y_%m_%d')
parmfiles = parmfiles[parmfiles.date > ed]

# Perform analysis for each unique DAY, not file. So, group parmfiles coming from same date.
options = {'pupil': True, 'rasterfs': 100}
pupil = []
R_HR = []
NR_HR = []
LI = []
for date in parmfiles['date'].unique():
    files = [p for i, p in parmfiles.iterrows() if p.date==date]
    files = [f['resppath']+f['parmfile'] for f in files]

    manager = BAPHYExperiment(files)
    rec = manager.get_recording(**options)

    # get R / NR hit rate and pupil size over valid trials

    # compute HR's over sliding window in baphy trial time
    # only on valid trials
    totalTrials = rec['pupil'].extract_epoch('TRIAL').shape[0]
    rec = rec.and_mask('INVALID_BAPHY_TRIAL', invert=True)
    validTrials = rec['pupil'].extract_epoch('TRIAL', mask=rec['mask']).shape[0]

    # find valid trial numbers using rec epochs times
    invalid_time = rec['pupil'].get_epoch_bounds('INVALID_BAPHY_TRIAL') 
    trial_epochs = rec['pupil'].get_epoch_bounds('TRIAL') 
    good_trials = [i+1 for i in range(0, trial_epochs.shape[0]) if trial_epochs[i] not in invalid_time]

    s = 0
    e = window_length
    nrhr = np.zeros(len(good_trials))
    rhr = np.zeros(len(good_trials))
    FAR = np.zeros(len(good_trials))
    params = manager.get_baphy_exptparams()[0]
    targets = np.array(params['TrialObject'][1]['TargetHandle'][1]['Names'])
    pump_dur = np.array(params['BehaveObject'][1]['PumpDuration'])
    r_tar = targets[pump_dur>0]
    nr_tar = targets[pump_dur==0]
    for i in range(0, len(good_trials)):
        print("{0} / {1}".format(i, len(good_trials)))
        trials = good_trials[s:e]
        out = manager.get_behavior_performance(trials=trials, **options)
        nrhr[i] = out['RR'][nr_tar[0]]
        rhr[i] = out['RR'][r_tar[0]]
        FAR[i] = out['RR']['Reference']
        
        if (len(good_trials) - i) < window_length:
            e += 1
        else:
            s += 1
            e += 1

    NR_HR.append(nrhr)
    R_HR.append(rhr)
    LI.append((rhr - nrhr) / (rhr + nrhr))

    peak_pupil = rec['pupil']._data.max() 
    rec_valid = rec.and_mask('INVALID_BAPHY_TRIAL', invert=True)
    p = rec['pupil'].extract_epoch('TRIAL', mask=rec_valid['mask'])
    p = np.nanmean(p, axis=-1).squeeze() / peak_pupil

    pupil.append(p)

f, ax = plt.subplots(1, 1)
for li, p in zip(LI, pupil):
    ax.plot(p, li, 'o')


plt.show()