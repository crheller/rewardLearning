"""
Load all Cordy BVT files with pupil (gDataRaw.eyewin=2). Calculate LI over sliding window. Plot LI vs. pupil size.
Normalize pupil size per day.
"""
import nems.db as nd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from nems_lbhb.baphy_experiment import BAPHYExperiment
import scipy.ndimage.filters as sf

window_length = 15 # in baphy trials
fpath = '/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/figures/'
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
options = {'pupil': True, 'rasterfs': 100}
window_length = 20

for date in parmfiles.date.unique():
    files = [p for i, p in parmfiles.iterrows() if p.date==date]
    files = [f['resppath']+f['parmfile'] for f in files]

    manager = BAPHYExperiment(files)
    rec = manager.get_recording(**options)


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
    HR = np.zeros(len(good_trials))
    HR2 = np.zeros(len(good_trials))
    FAR = np.zeros(len(good_trials))
    for i in range(0, len(good_trials)):
        print("{0} / {1}".format(i, len(good_trials)))
        trials = good_trials[s:e]
        out = manager.get_behavior_performance(trials=trials, **options)
        targets = [t for t in out['RR'].keys() if 'Reference' not in t]
        HR[i] = out['RR'][targets[0]]
        HR2[i] = out['RR'][targets[1]]
        FAR[i] = out['RR']['Reference']
        
        if (len(good_trials) - i) < window_length:
            e += 1
        else:
            s += 1
            e += 1

    f, ax = plt.subplots(2, 1)

    ax[0].plot(good_trials, HR, '.-', label=targets[0])
    ax[0].plot(good_trials, HR2, '.-', label=targets[1])
    ax[0].plot(good_trials, FAR, '.-', label='FAR')

    ax[0].set_ylim((0, 1.1))
    ax[0].set_ylabel('Hit rate')
    ax[0].set_xlabel('Baphy Trials')
    ax[0].legend()

    # get mean pupil per valid baphy trial
    rec_valid = rec.and_mask('INVALID_BAPHY_TRIAL', invert=True)
    pupil = rec['pupil'].extract_epoch('TRIAL', mask=rec_valid['mask'])
    pupil = np.nanmean(pupil, axis=-1).squeeze()
    ax[1].plot(good_trials, sf.gaussian_filter1d(pupil, 2), color='purple')
    #ax[1].plot(sf.gaussian_filter1d(rec_valid.apply_mask()['pupil']._data.squeeze(), sigma), color='purple')
    ax[1].set_ylabel('pupil')
    ax[1].set_xlabel('Baphy Trials')

    f.tight_layout()

    f.savefig(fpath + parmfiles[parmfiles.date==date]['parmfile'].values[0] + '.png')

plt.show()