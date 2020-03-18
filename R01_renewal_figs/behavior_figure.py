"""
1) Example behavioral session, show divisions between file blocks on that day.
    Idea would be that behavior gets better over the course of the day
2) Summary of learning index for both animals
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nems.db as nd
import datetime as dt
from nems_lbhb.baphy_experiment import BAPHYExperiment
import scipy.stats as ss
import matplotlib as mpl
mpl.rcParams.update({'svg.fonttype': 'none'})

# set up axes
plt.figure(figsize=(14, 4))
example_ax = plt.subplot2grid((1, 6), (0, 0), colspan=4)
crd_ax = plt.subplot2grid((1, 6), (0, 4), colspan=1)
drx_ax = plt.subplot2grid((1, 6), (0, 5), colspan=1)
median = True

# ========================================= EXAMPLE SESSION ===========================================================
# define example session
filename="Cordyceps_2020_02_20"
datestr = "2020-02-20"
window_length = 25

# get relevant parmfiles
sql = "SELECT parmfile, resppath, gDataRaw.id, cellid, gData.svalue, gData.value FROM gDataRaw INNER JOIN gData on (gDataRaw.id=gData.rawid and gData.name='Tar_Frequencies') WHERE runclass=%s and behavior='active' and parmfile like %s"
parmfiles1 = nd.pd_query(sql, ('BVT', filename+"%"))
parmfiles1['date'] = dt.datetime.strptime(datestr, '%Y-%m-%d')
parmfiles1 = parmfiles1.set_index('id')

sql = "SELECT parmfile, resppath, gDataRaw.id, cellid, gData.svalue FROM gDataRaw INNER JOIN gData on (gDataRaw.id=gData.rawid and gData.name='Behave_PumpDuration') WHERE runclass=%s and behavior='active' and parmfile like %s"
parmfiles = nd.pd_query(sql, ('BVT', filename+"%"))
parmfiles['date'] = dt.datetime.strptime(datestr, '%Y-%m-%d')
parmfiles = parmfiles.set_index('id')
parmfiles = parmfiles.rename(columns={'svalue': 'pumpdur'})

# now join on rawid
parmfiles = pd.concat([parmfiles['pumpdur'], parmfiles1], axis=1, join='outer')

parmfiles.loc[parmfiles.svalue.isnull(),'svalue']=parmfiles.loc[parmfiles.svalue.isnull(),'value'].astype(str)
parmfiles['svalue'] = parmfiles['svalue'].str.strip('[]')
parmfiles['pumpdur'] = parmfiles['pumpdur'].str.strip('[]')
parmfiles['svalue']=parmfiles['svalue'].str.replace('\.0','')
parmfiles['svalue']=parmfiles['svalue'].str.replace(' ', '+')
parmfiles['pumpdur']=parmfiles['pumpdur'].str.replace('\.0','')
parmfiles['pumpdur']=parmfiles['pumpdur'].str.replace(' ', '+')

parmfiles['session']=parmfiles.date.dt.strftime("%Y-%m-%d")+"_"+parmfiles['svalue']+"_"+parmfiles['pumpdur']

files = [r+f for r, f in zip(parmfiles['resppath'], parmfiles['parmfile'])]

# get baphy manager
manager = BAPHYExperiment(files)
options = {'pupil': False, 'rasterfs': 100}
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
file_divs = good_trials[np.argwhere(np.diff(rec['fileidx'].extract_epoch('TRIAL', mask=rec['mask'])[:,0,0])).squeeze() - window_length]

step = 2
trialidx = np.arange(0, len(good_trials)-window_length+step, step)
trialcount = len(trialidx)
HR = np.zeros(trialcount)
HR2 = np.zeros(trialcount)
FAR = np.zeros(trialcount)
LI = np.zeros(trialcount)

# find rewarded target using the unique session id
session = parmfiles['session'].iloc[0]
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


tt = np.array(good_trials)[trialidx]
example_ax.plot(tt, HR, color='limegreen', lw=2, marker='o', label='R. HR')
example_ax.plot(tt, HR2, color='grey', lw=2, marker='o', label='N.R. HR')
example_ax.plot(tt, FAR, color='red', lw=2, marker='o', label='FAR')
example_ax.plot(tt, LI, color='k', lw=2, linestyle='--', label='Learning Index')

example_ax.legend(frameon=False)
example_ax.set_ylim((-0.1, 1.1))
example_ax.set_xlabel('Trials')
example_ax.set_ylabel('HitRate / LI')

# =================================== LI SUMMARY FOR EACH ANIMAL ===========================================
# load summary of results for each animal
crd = pd.read_pickle('/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/results/Cordyceps_strictSlidingWindow.pickle')
drx = pd.read_pickle('/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/results/Drechsler_strictSlidingWindow.pickle')
li_metric = 'LI'  # LI_far, LI_di, LI
min_tar = 5  # number of targets per category to qualify
min_HR = 0.3  # mean hit rate across targets (weighted by number of presentations) must be at least 


# ============ CRD ============
# split data in half. Save max li in first half and max in second half
# if the window meet above criteria
#   LI early/late
#   Ratio of nTar trials early/late
df = pd.DataFrame(index=crd.session.unique(), columns=['e_LI', 'l_LI', 'e_nR', 'l_nR', 'e_nNR', 'l_nNR', 'window_diff'])
for session in crd.session.unique():
    data = crd[crd.session==session]
    
    early_cut = int(data.shape[0] / 2.5)
    late_cut = early_cut
    late_cut_end = data.shape[0]
    # find early window LI max
    early_li_max = -1
    for i in range(0, early_cut):
        d = data.iloc[i]
        tot = d['n_R'] + d['n_NR']
        wHR = ((d['n_R'] / tot) * d['HR']) + ((d['n_NR'] / tot) * d['HR2']) / 2
        if (d['n_R'] >= min_tar) & (d['n_NR'] >= min_tar) & (wHR >= min_HR):
            li = data.iloc[i][li_metric]
            if li > early_li_max:
                early_li_max = li

    # find late window
    late_li_max = -1
    for i in range(late_cut, late_cut_end):
        d = data.iloc[i]
        tot = d['n_R'] + d['n_NR']
        wHR = ((d['n_R'] / tot) * d['HR']) + ((d['n_NR'] / tot) * d['HR2']) / 2
        if (d['n_R'] >= min_tar) & (d['n_NR'] >= min_tar) & (wHR >= min_HR):
            li = data.iloc[i][li_metric]
            if li > late_li_max:
                late_li_max = li

    # get early stats
    df.loc[session]['e_LI'] = early_li_max

    # get late stats
    df.loc[session]['l_LI'] = late_li_max

if median:
    crd_ax.bar([0, 1], [df['e_LI'].median(), df['l_LI'].median()], edgecolor=['gold', 'purple'], color='none', lw=2)
else:
    crd_ax.bar([0, 1], [df['e_LI'].mean(), df['l_LI'].mean()], edgecolor=['gold', 'purple'], color='none', lw=2)
e_idx = np.random.normal(0, 0.06, df.shape[0])
l_idx = e_idx + 1
crd_ax.plot(e_idx, df['e_LI'], 'k.')
crd_ax.plot(l_idx, df['l_LI'], 'k.')
crd_ax.axhline(0, linestyle='--', color='k')
crd_ax.set_ylabel('Max LI')
crd_ax.set_xticks([0, 1])
crd_ax.set_xticklabels(['Early', 'Late'])
crd_ax.set_ylim((-.2, 1.1))
if median:
    early = np.round(df['e_LI'].median(), 3)
    late = np.round(df['l_LI'].median(), 3)
else:
    early = np.round(df['e_LI'].mean(), 3)
    late = np.round(df['l_LI'].mean(), 3)
pval = np.round(ss.wilcoxon(df['e_LI'], df['l_LI'])[1], 3)
crd_ax.set_title("Cordyceps \n early: {0}, late: {1} \n pval: {2}".format(early, late, pval))

# ============ DRX ============
# split data in half. Save max li in first half and max in second half
# if the window meet above criteria
#   LI early/late
#   Ratio of nTar trials early/late
df = pd.DataFrame(index=drx.session.unique(), columns=['e_LI', 'l_LI', 'e_nR', 'l_nR', 'e_nNR', 'l_nNR', 'window_diff'])
for session in drx.session.unique():
    data = drx[drx.session==session]
    
    early_cut = int(data.shape[0] / 2.5)
    late_cut = early_cut
    late_cut_end = data.shape[0]
    # find early window LI max
    early_li_max = -1
    for i in range(0, early_cut):
        d = data.iloc[i]
        tot = d['n_R'] + d['n_NR']
        wHR = ((d['n_R'] / tot) * d['HR']) + ((d['n_NR'] / tot) * d['HR2']) / 2
        if (d['n_R'] >= min_tar) & (d['n_NR'] >= min_tar) & (wHR >= min_HR):
            li = data.iloc[i][li_metric]
            if li > early_li_max:
                early_li_max = li

    # find late window
    late_li_max = -1
    for i in range(late_cut, late_cut_end):
        d = data.iloc[i]
        tot = d['n_R'] + d['n_NR']
        wHR = ((d['n_R'] / tot) * d['HR']) + ((d['n_NR'] / tot) * d['HR2']) / 2
        if (d['n_R'] >= min_tar) & (d['n_NR'] >= min_tar) & (wHR >= min_HR):
            li = data.iloc[i][li_metric]
            if li > late_li_max:
                late_li_max = li

    # get early stats
    df.loc[session]['e_LI'] = early_li_max

    # get late stats
    df.loc[session]['l_LI'] = late_li_max


if median:
    drx_ax.bar([0, 1], [df['e_LI'].median(), df['l_LI'].median()], edgecolor=['gold', 'purple'], color='none', lw=2)
else:   
    drx_ax.bar([0, 1], [df['e_LI'].mean(), df['l_LI'].mean()], edgecolor=['gold', 'purple'], color='none', lw=2)
e_idx = np.random.normal(0, 0.06, df.shape[0])
l_idx = e_idx + 1
drx_ax.plot(e_idx, df['e_LI'], 'k.')
drx_ax.plot(l_idx, df['l_LI'], 'k.')
drx_ax.axhline(0, linestyle='--', color='k')
drx_ax.set_ylabel('Max LI')
drx_ax.set_xticks([0, 1])
drx_ax.set_xticklabels(['Early', 'Late'])
drx_ax.set_ylim((-.2, 1.1))
if median:
    early = np.round(df['e_LI'].median(), 3)
    late = np.round(df['l_LI'].median(), 3)
else:
    early = np.round(df['e_LI'].mean(), 3)
    late = np.round(df['l_LI'].mean(), 3)
pval = np.round(ss.wilcoxon(df['e_LI'], df['l_LI'])[1], 3)
drx_ax.set_title("Drechsler \n early: {0}, late: {1} \n pval: {2}".format(early, late, pval))

plt.tight_layout()

plt.show()
