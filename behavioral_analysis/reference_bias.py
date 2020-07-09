"""
Determine if the False Alarms are biased towards the noise centered on the target frequency.
"""
import nems_lbhb.behavior as beh
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import nems_lbhb.io as io
import nems.db as nd
from nems_lbhb.baphy_experiment import BAPHYExperiment
import os

def get_RR(rec, epoch='TARGET', response_window=(0.2, 1)):
    """
    Get response rate (RR) to given epoch. 
        e.g. RR = num. times lick occured in response_window following epoch onset / total epoch presentations
    """
    count = sum(rec.epochs.name==epoch)
    bounds = rec['fileidx'].get_epoch_bounds(epoch)
    resp_bounds = np.stack([bounds[:, 0]+response_window[0], bounds[:, 0]+response_window[1]]).T
    lick_times = rec['fileidx'].get_epoch_bounds('LICK')[:, 0]

    hits = 0
    for r in range(resp_bounds.shape[0]):
        if np.any((lick_times >= resp_bounds[r][0]) & (lick_times <= resp_bounds[r][1])):
            hits += 1
        else:
            pass

    return hits / count

fpath = '/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/figures/'
dfpath = '/auto/users/hellerc/code/projects/rewardLearning/R01_renewal_figs/results/'
runclass = 'BVT'
animal = 'Cordyceps'
earliest_date = '2019_11_09'

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

# get rid of nan pumpdur rows (those are passive files)
parmfiles = parmfiles[~pd.isna(parmfiles['pumpdur'])]

# unique id for each session based on the target frequenicies and reward values
parmfiles.loc[parmfiles.svalue.isnull(),'svalue']=parmfiles.loc[parmfiles.svalue.isnull(),'value'].astype(str)
parmfiles['svalue'] = parmfiles['svalue'].str.strip('[]')
parmfiles['pumpdur'] = parmfiles['pumpdur'].str.strip('[]')
parmfiles['svalue']=parmfiles['svalue'].str.replace('\.0','')
parmfiles['svalue']=parmfiles['svalue'].str.replace(' ', '+')
parmfiles['pumpdur']=parmfiles['pumpdur'].str.replace('\.0','')
parmfiles['pumpdur']=parmfiles['pumpdur'].str.replace(' ', '+')

parmfiles['session']=parmfiles.date.dt.strftime("%Y-%m-%d")+"_"+parmfiles['svalue']+"_"+parmfiles['pumpdur']

r_center = np.nan * np.ones((len(parmfiles.session.unique()), 30))
nr_center = np.nan * np.ones((len(parmfiles.session.unique()), 30))
raw =  np.nan * np.ones((len(parmfiles.session.unique()), 15))

# loop over days
for i, session in enumerate(parmfiles.session.unique()):
    files = parmfiles[parmfiles.session==session]
    date = files.iloc[0].date
    files = [r + p for r, p in zip(files['resppath'], files['parmfile'])]

    manager = BAPHYExperiment(files)
    rec = manager.get_recording(**{'rasterfs': 100})

    # get rewarded target frequencies (and non-rewarded)
    params = manager.get_baphy_exptparams()[0]
    PumpDur = np.array(params['BehaveObject'][1]['PumpDuration'])
    TargetFreq = np.array(params['TrialObject'][1]['TargetHandle'][1]['Names'])
    RTarget = TargetFreq[PumpDur>0]
    RTargetStr = [str(r) for r in RTarget]
    NRTarget = TargetFreq[PumpDur==0]
    NRTargetStr = [str(r) for r in NRTarget]

    # get list of reference stimuli
    ref_stims = [e for e in rec.epochs.name.unique() if 'STIM_' in e]
    cfs = [int(e.replace('STIM_', '')) for e in ref_stims]
    idx = np.argsort(cfs)
    cfs = np.array(cfs)[idx]
    ref_stims = np.array(ref_stims)[idx]

    # get hit rate for each reference
    rr = np.zeros(len(ref_stims))
    for i, r in enumerate(ref_stims):
        rr[i] = get_RR(rec, epoch=r, response_window=(0.1, 1))

    
    r_ref = cfs.flat[np.abs(cfs - int(RTarget[0])).argmin()]
    nr_ref = cfs.flat[np.abs(cfs - int(NRTarget[0])).argmin()]

    if len(ref_stims) == 16:
        # center on rewarded
        ridx = np.argwhere(cfs==r_ref)[0][0]
        if ridx == 16:
            ridx = 15
            rr = rr[1:]
        else:
            rr = rr[:15]
        r_center[i][(15-ridx):(15+(15-ridx))] = rr

        # center on non-rewarded
        nridx = np.argwhere(cfs==nr_ref)[0][0]
        if nridx == 16:
            nridx = 15
            rr = rr[1:]
        else:
            rr = rr[:15]
        nr_center[i][(15-nridx):(15+(15-nridx))] = rr
    elif len(ref_stims) == 15:
        # center on rewarded
        ridx = np.argwhere(cfs==r_ref)[0][0]
        r_center[i][(15-ridx):(15+(15-ridx))] = rr

        # center on non-rewarded
        nridx = np.argwhere(cfs==nr_ref)[0][0]
        nr_center[i][(15-nridx):(15+(15-nridx))] = rr
    else:
        raise ValueError("unknown case")

    # save raw
    raw[i] = rr


# plot raw results, rewarded centered, and not-rewarded centered

f, ax = plt.subplots(1, 3, figsize=(9, 3))

m = np.nanmean(raw, axis=0)
sd = np.nanstd(raw, axis=0)
ax[0].plot(cfs, m, lw=2)
ax[0].fill_between(cfs, m-sd, m+sd, alpha=0.2)
ax[0].set_ylabel('FA Rate')
ax[0].set_xlabel('Center freq')
ax[0].set_title('Raw FA Rate')
ax[0].set_xscale('log', basex=2)

oct_range = np.linspace(-0.3 * 15, 0.3 * 15, 30)

m = np.nanmean(r_center, axis=0)
sd = np.nanstd(r_center, axis=0)
ax[1].plot(oct_range, m, lw=2)
ax[1].fill_between(oct_range, m-sd, m+sd, alpha=0.2)
ax[1].set_ylabel('FA Rate')
ax[1].set_xlabel('Octaves from rewarded')
ax[1].set_title('FA Rate Centered on Rewarded freq')

m = np.nanmean(nr_center, axis=0)
sd = np.nanstd(nr_center, axis=0)
ax[2].plot(oct_range, m, lw=2)
ax[2].fill_between(oct_range, m-sd, m+sd, alpha=0.2)
ax[2].set_ylabel('FA Rate')
ax[2].set_xlabel('Octaves from Non-rewarded')
ax[2].set_title('FA Rate Centered on Non-Rewarded freq')


f.tight_layout()

plt.show()

