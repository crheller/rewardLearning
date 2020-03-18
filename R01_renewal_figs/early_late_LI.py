"""
Cordy early / late LI analysis.
Compare target preference at the end of the day vs. the beginnig of the day for 
days where only one set of targets was used (started on 11/22/2019)
"""
import nems.db as nd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nems_lbhb.baphy_experiment import BAPHYExperiment

runclass = 'BVT'
animal = 'Cordyceps'
an_regex = "%"+animal+"%"
min_trials = 50
earliest_date = '2019_11_22'
window_length = 50 # how many trials in beginning / end to include

# get list of all training parmfiles
sql = "SELECT parmfile, resppath FROM gDataRaw WHERE runclass=%s and resppath like %s and training = 1 and bad=0 and trials>%s and behavior='active'"
parmfiles = nd.pd_query(sql, (runclass, an_regex, min_trials))

# screen for dates
parmfiles['date'] = [dt.datetime.strptime('-'.join(x.split('_')[1:-2]), '%Y-%m-%d') for x in parmfiles.parmfile]
ed = dt.datetime.strptime(earliest_date, '%Y_%m_%d')
parmfiles = parmfiles[parmfiles.date > ed]

results = pd.DataFrame(index=parmfiles['date'].unique(), columns=['Early', 'Late', 'Overall', 'nTrials'])
options = {}
for date in parmfiles['date'].unique():
    files = [p for i, p in parmfiles.iterrows() if p.date==date]
    files = [f['resppath']+f['parmfile'] for f in files]

    manager = BAPHYExperiment(files)
    events = manager.get_behavior_events(correction_method='baphy', **options)
    events = manager._stack_events(events)

    good_trials = events[(~events.invalidTrial)].Trial.unique()
    # note, number of target trials in performance dict doesn't have to add up to
    # number of window_length trials. There could (definitely are) FA trials
    # that are "valid trials", but not counted for behavior analysis bc target never
    # played

    params = manager.get_baphy_exptparams()[0]
    targets = np.array(params['TrialObject'][1]['TargetHandle'][1]['Names'])
    pump_dur = np.array(params['BehaveObject'][1]['PumpDuration'])
    r_tar = targets[pump_dur>0][0]
    nr_tar = targets[pump_dur==0][0]

    if len(good_trials) < (window_length * 1.5):
        all_trials = manager.get_behavior_performance(trials=None, **options)
        results.loc[date]['Overall'] = (all_trials['RR'][r_tar] - all_trials['RR'][nr_tar]) / \
                                    (all_trials['RR'][r_tar] + all_trials['RR'][nr_tar])
    else:
        all_trials = manager.get_behavior_performance(trials=None, **options)
        results.loc[date]['Overall'] = (all_trials['RR'][r_tar] - all_trials['RR'][nr_tar]) / \
                                            (all_trials['RR'][r_tar] + all_trials['RR'][nr_tar])
        results.loc[date]['nTrials'] = len(good_trials)

        # get LI for first 50 trials
        early_trials = good_trials[:window_length]
        early = manager.get_behavior_performance(trials=early_trials, **options)

        # get LI for last 50 trials
        late_trials = good_trials[-window_length:-10]
        late = manager.get_behavior_performance(trials=late_trials, **options)

        results.loc[date]['Early'] = (early['RR'][r_tar] - early['RR'][nr_tar]) / (early['RR'][r_tar] + early['RR'][nr_tar])
        results.loc[date]['Late'] = (late['RR'][r_tar] - late['RR'][nr_tar]) / (late['RR'][r_tar] + late['RR'][nr_tar])

# compare early / late learning index and show overall distribution of LI
f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].bar([0, 1], [results['Early'].mean(), results['Late'].mean()], 
                yerr=[results['Early'].sem(), results['Late'].sem()],
                color='lightgrey', edgecolor='k')
ax[0].set_ylabel('Learning Index')
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['Early', 'Late'])

bins = np.arange(-.5, .5, 0.05)
results['Overall'].plot.hist(edgecolor='white', bins=bins, ax=ax[1])
ax[1].set_xlabel('Learning Index')
ax[1].set_title('All Valid Trials')

f.tight_layout()

plt.show()