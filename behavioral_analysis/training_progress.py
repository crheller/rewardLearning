import nems.db as nd
import nems_lbhb.behavior as beh
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotting as cplt
from scipy.optimize import curve_fit

runclass = 'BVT'
animal = 'Drechsler'
an_regex = "%"+animal+"%"
min_trials = 50
earliest_date = '2019_10_03'  # date when we started repeating unrewarded trials
earliest_date = '2019_11_03'
valid = True
verbose = True
figpath = '/auto/users/hellerc/code/projects/rewardLearning/behavioral_analysis/plots/'
# get list of all training parmfiles
sql = "SELECT parmfile, resppath FROM gDataRaw WHERE runclass=%s and resppath like %s and training = 1 and bad=0 and trials>%s and behavior='active'"
parmfiles = nd.pd_query(sql, (runclass, an_regex, min_trials))

# screen for dates
parmfiles['date'] = [dt.datetime.strptime('-'.join(x.split('_')[1:-2]), '%Y-%m-%d') for x in parmfiles.parmfile]
ed = dt.datetime.strptime(earliest_date, '%Y_%m_%d')
parmfiles = parmfiles[parmfiles.date > ed]

# for each day, compute relevant stats / save per-day plots
days = np.unique(parmfiles['date'])
cols = ['DI_rew', 'DI_nrew']
results_df = pd.DataFrame(index=days, columns=cols)
block_results_df = pd.DataFrame(index=parmfiles['date'], columns=cols+['rew_tar', 'nrew_tar'])
for i, day in enumerate(days):
    print('analyzing all blocks on day: {0}'.format(str(day)))
    rdi = []
    nrdi = []

    # make figure for this day
    rows = int((parmfiles.date==day).sum())
    f, ax = plt.subplots(rows, 2, figsize=(16, 12))
    f.canvas.set_window_title(day)
    for j, (pf, rp) in enumerate(zip(parmfiles[parmfiles.date==day]['parmfile'], parmfiles[parmfiles.date==day]['resppath'])):
        if rows == 1:
            j = None
        try:
            exptparams, exptevents = beh.load_behavior(rp+pf, classify_trials=True)
        except:
            continue

        if valid:
            options = {'keep_early_trials': False, 'keep_cue_trials': False, 'keep_following_incorrect_trial': False}
        else:
            options = {'keep_early_trials': True, 'keep_cue_trials': True, 'keep_following_incorrect_trial': True}

        metrics = beh.compute_metrics(exptparams, exptevents, **options)

        targets = list(metrics['DI'].keys())

        pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])
        rew_idx = [True if i>0 else False for i in pump_dur]
        rew_tar = np.array(targets)[rew_idx]

        for k in targets:
            if k in rew_tar:
                rdi.append(metrics['DI'][k])
            else:
                nrdi.append(metrics['DI'][k])

        if j is None:
            ax1 = ax[j, 0][0]
            ax2 = ax[j, 1][0]
            block_results_df.loc[day][cols[0]] = np.nanmean(rdi)
            block_results_df.loc[day][cols[1]] = np.nanmean(nrdi)
            rtars = [int(t) for t in rew_tar]
            nrtars = [int(t) for t in targets if t not in rew_tar]
            block_results_df.loc[day]['rew_tar'] = np.nanmean(rtars)
            block_results_df.loc[day]['nrew_tar'] = np.nanmean(nrtars)
        else:
            ax1 = ax[j, 0]
            ax2 = ax[j, 1]
            block_results_df.loc[day][cols[0]][j] = np.nanmean(rdi)
            block_results_df.loc[day][cols[1]][j] = np.nanmean(nrdi)
            rtars = [int(t) for t in rew_tar]
            nrtars = [int(t) for t in targets if t not in rew_tar]
            block_results_df.loc[day]['rew_tar'][j] = np.nanmean(rtars)
            block_results_df.loc[day]['nrew_tar'][j] = np.nanmean(nrtars)



        # plot RT histograms overall data
        RTs = beh.compute_RTs(exptparams, exptevents, **options)
        pump_dur = exptparams['BehaveObject'][1]['PumpDuration']
        pump_dur.append(0)

        bins = np.arange(0, 1, 0.05)
        for i, k in enumerate(RTs.keys()):
            counts, xvals = np.histogram(RTs[k], bins=bins)
            rwdstr = 'rew.' if pump_dur[i] > 0 else ''

            try:
                DI = round(metrics['DI'][k], 2)
            except:
                DI = 'N/A'
            if k != 'Reference':
                di = round(metrics['DI'][k],3)
            else:
                di = None
            ax1.step(xvals[:-1], np.cumsum(counts) / len(RTs[k]) * metrics['RR'][k], label='{0}, DI: {1}'.format(rwdstr, di))

        ax1.set_ylim((0, 1))
        ax1.legend(fontsize=6)
        ax1.set_xlabel('Reaction time', fontsize=8)
        ax1.set_ylabel('HR', fontsize=8)
        ax1.set_title("All trials", fontsize=8)
        ax1.set_aspect(cplt.get_square_asp(ax1))

        # plot DI
        di = []
        for i, t in enumerate(targets):
            n = metrics['nTrials'][t]
            di.append(metrics['DI'][t])
            ax2.plot(i, metrics['DI'][t], 'o', label='{0}, n: {1}'.format(t, n))
        ax2.plot(di, 'k-')
        ax2.legend(fontsize=8)
        ax2.set_ylabel('DI')
        ax2.set_ylim((0, 1))
        ax2.set_aspect(cplt.get_square_asp(ax2))

        f.tight_layout()

        day_string = str(day).split('T')[0]
        if valid:
            day_string += '_validTrials'
        else:
            day_string += '_allTrials'
        f.savefig(figpath+day_string+'.png')

    results_df.loc[day][cols] = [np.mean(rdi), np.nanmean(nrdi)]

if not verbose:
    plt.close('all')

# plot by day
f, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(results_df['DI_rew'].values, 'o-', color='blue', label='rewarded')
ax[0].plot(results_df['DI_nrew'].values, 'o-', color='red', label='non-rewarded')
ax[0].axhline(0.5, linestyle='--', color='grey')
ax[0].set_ylabel('DI')
ax[0].set_xlabel('Date')
ax[0].legend()

ax[1].plot((results_df['DI_rew'].values - results_df['DI_nrew'].values) / \
        (results_df['DI_rew'].values + results_df['DI_nrew'].values), 'o-', color='k')
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].set_xticks(np.arange(0, results_df.shape[0]))
ax[1].set_xticklabels([str(x).split(' ')[0] for x in results_df.index], rotation=45)
ax[1].set_ylabel('Rew. pref.')
ax[1].set_xlabel('Date')


f.tight_layout()


# plot by block
f, ax = plt.subplots(3, 1, sharex=True)

ax[0].plot(block_results_df['DI_rew'].values, 'o-', color='blue', label='rewarded')
ax[0].plot(block_results_df['DI_nrew'].values, 'o-', color='red', label='non-rewarded')
ax[0].axhline(0.5, linestyle='--', color='grey')
ax[0].set_ylabel('DI')
ax[0].set_xlabel('Date')
ax[0].legend()

ax[1].plot(block_results_df['rew_tar'].values, 'o-', color='blue', label='rewarded freq')
ax[1].plot(block_results_df['nrew_tar'].values, 'o-', color='red', label='non-rewarded')
ax[1].set_yscale('log')
ax[1].set_ylabel('freq')
ax[1].legend(fontsize=6)

ax[2].plot((block_results_df['DI_rew'].values - block_results_df['DI_nrew'].values) / \
        (block_results_df['DI_rew'].values + block_results_df['DI_nrew'].values), 'o-', color='k')
ax[2].axhline(0, linestyle='--', color='grey')
ax[2].set_xticks(np.arange(0, block_results_df.shape[0]))
ax[2].set_xticklabels([str(x).split(' ')[0] for x in block_results_df.index], rotation=90)
ax[2].set_ylabel('Rew. pref.')
ax[2].set_xlabel('Date')

fn = 'block_summary'
if valid:
    fn += '_validTrials'
else:
    fn += '_allTrials'
f.savefig(figpath+fn+'.png')

f.tight_layout()

# separate based on reward association and plot mean rew vs. non rew. DI
mask1 = block_results_df['rew_tar'] > block_results_df['nrew_tar']
mask2 = block_results_df['rew_tar'] < block_results_df['nrew_tar']
block_results_df['octave_sep'] = block_results_df['rew_tar'] / block_results_df['nrew_tar']
block_results_df['octave_sep'][mask2] = block_results_df['nrew_tar'] / block_results_df['rew_tar']
block_results_df['reversal'] = [False] + [True if mask1[i] != mask1[i-1] else False for i in range(1, mask1.shape[0])]

rdi_hf = block_results_df[mask1]['DI_rew'].mean()
nrdi_hf =block_results_df[mask1]['DI_nrew'].mean()
rdi_hf_sem = block_results_df[mask1]['DI_rew'].sem()
nrdi_hf_sem =block_results_df[mask1]['DI_nrew'].sem()

rdi_lf = block_results_df[mask2]['DI_rew'].mean()
nrdi_lf =block_results_df[mask2]['DI_nrew'].mean()
rdi_lf_sem = block_results_df[mask2]['DI_rew'].sem()
nrdi_lf_sem =block_results_df[mask2]['DI_nrew'].sem()

f, ax = plt.subplots(1, 2)

ax[0].set_title('all blocks')
ax[0].errorbar([0, 1], [rdi_hf, nrdi_hf], yerr=[rdi_hf_sem, nrdi_hf_sem], color='grey', label='high freq. rewarded')
ax[0].errorbar([0, 1], [rdi_lf, nrdi_lf], yerr=[rdi_lf_sem, nrdi_lf_sem], color='black', label='low freq. rewarded')
ax[0].legend(fontsize=8)
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['rewarded', 'unrewarded'])
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].set_title('exclude reversal blocks')
m = block_results_df['reversal']
rdi_hf = block_results_df[mask1 & m]['DI_rew'].mean()
nrdi_hf =block_results_df[mask1 & m]['DI_nrew'].mean()
rdi_hf_sem = block_results_df[mask1 & m]['DI_rew'].sem()
nrdi_hf_sem =block_results_df[mask1 & m]['DI_nrew'].sem()

rdi_lf = block_results_df[mask2 & m]['DI_rew'].mean()
nrdi_lf =block_results_df[mask2 & m]['DI_nrew'].mean()
rdi_lf_sem = block_results_df[mask2 & m]['DI_rew'].sem()
nrdi_lf_sem =block_results_df[mask2 & m]['DI_nrew'].sem()

ax[1].errorbar([0, 1], [rdi_hf, nrdi_hf], yerr=[rdi_hf_sem, nrdi_hf_sem], color='grey', label='high freq. rewarded')
ax[1].errorbar([0, 1], [rdi_lf, nrdi_lf], yerr=[rdi_lf_sem, nrdi_lf_sem], color='black', label='low freq. rewarded')
ax[1].legend(fontsize=8)
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(['rewarded', 'unrewarded'])
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

# plot as function of octave separation between rew. / un. rew.

block_results_df = block_results_df.sort_values(by='octave_sep')
oct_range = np.arange(0, np.max(block_results_df['octave_sep']+1), 0.1)

rew_pref = (block_results_df['DI_rew'] - block_results_df['DI_nrew']) / \
            (block_results_df['DI_rew'] + block_results_df['DI_nrew'])

f, ax = plt.subplots(1, 2)
ax[0].plot(block_results_df['octave_sep'], rew_pref, 'ko')
ax[0].set_xlim((0, 10.5))
ax[0].set_ylabel('Rew. Preference')
ax[0].set_xlabel('sepatation (octaves)')
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].plot(block_results_df['octave_sep'], block_results_df['DI_rew'], 'bo', label='rewarded')
ax[1].plot(block_results_df['octave_sep'], block_results_df['DI_nrew'], 'ro', label='non-rewarded')
ax[1].set_xlim((0, 10.5))
ax[1].set_ylabel('DI')
ax[1].set_xlabel('sepatation (octaves)')
ax[1].axhline(0.5, linestyle='--', color='grey')
ax[1].legend(fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

plt.show()
