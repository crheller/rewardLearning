from nems_lbhb.baphy_experiment import BAPHYExperiment
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import charlieTools.baphy_remote as br
from charlieTools.plotting import compute_ellipse
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
data_path = '/auto/data/daq/'

animal = 'Cordyceps'
site = 'CRD012b' #CRD009b, CRD010b, CRD011c, CRD012b
runclass = 'TBP'
run_nums = [12, 13]  # [5, 6, 7, 8], [5, 6, 7, 9], [6, 7, 8], [12, 13]

options = {'rasterfs': 20, 'pupil': False, 'resp': True}

path = os.path.join(data_path, animal, site[:-1])
parmfiles = [os.path.join(path, f) for f in os.listdir(path) if \
                                            (runclass in f) & (f.endswith('.m') & (site in f))]
parmfiles = [f for f in parmfiles if (int(os.path.basename(f)[7:9]) in run_nums)]

manager = BAPHYExperiment(parmfile=parmfiles)
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()

files = [f for f in rec['resp'].epochs.name.unique() if 'FILE_' in f]
targets = [f for f in rec['resp'].epochs.name.unique() if 'TAR_' in f]
catch = [f for f in rec['resp'].epochs.name.unique() if 'CAT_' in f]

sounds = np.array(targets + catch)
sounds = [str(s) for s in sounds[np.argsort([int(s[-1]) for s in sounds])]]
ref_stims = [x for x in rec['resp'].epochs.name.unique() if 'STIM_' in x]
idx = np.argsort([int(s.split('_')[-1]) for s in ref_stims])
ref_stims = np.array(ref_stims)[idx].tolist()
all_stims = ref_stims + sounds

# ================================================================================================
# Plot "raw" data -- tuning curves / psth's . Get sense of things

# PSTHs for REFs
ref_dur = int(manager.get_baphy_exptparams()[0]['TrialObject'][1]['ReferenceHandle'][1]['Duration'] * options['rasterfs'])
pre_post = int(manager.get_baphy_exptparams()[0]['TrialObject'][1]['ReferenceHandle'][1]['PostStimSilence'] * options['rasterfs'])
d1 = int(np.sqrt(rec['resp'].shape[0]))
f, ax = plt.subplots(d1+1, d1+1, figsize=(16, 12))
for i in range(rec['resp'].shape[0]):
    if i == 0:
        br.psth(rec, chan=rec['resp'].chans[i], epochs=ref_stims, ep_dur=ref_dur, cmap='viridis', prestim=pre_post, ax=ax.flatten()[i])
    else:
        br.psth(rec, chan=rec['resp'].chans[i], epochs=ref_stims, ep_dur=ref_dur, cmap='viridis', prestim=pre_post, supp_legend=True, ax=ax.flatten()[i])
f.tight_layout()


# PSTHs for TARs (and CATCH)

# TUNING CURVES
cfs = [s.split('_')[-1] for s in ref_stims]
prestim, poststim = int(options['rasterfs']*0.1), int(options['rasterfs']*0.3)
ftc_all = []
f, ax = plt.subplots(d1+1, d1+1, figsize=(16, 12))
for i in range(rec['resp'].shape[0]):
    d = rec['resp'].extract_channels([rec['resp'].chans[i]]).extract_epochs(ref_stims)
    ftc = [d[e][~np.isnan(d[e][:,0,prestim:poststim].sum(axis=-1)), :, prestim:poststim].mean() for e in d.keys()]
    ftc_all.append(ftc)
    if i == 0:
        ax.flatten()[i].plot(ftc)
    else:
        ax.flatten()[i].plot(ftc)
    
    ax.flatten()[i].set_xticks(range(len(ftc)))
    ax.flatten()[i].set_xticklabels(cfs, fontsize=6, rotation=45)
    ax.flatten()[i].set_xlabel('CF', fontsize=6)
    ax.flatten()[i].set_ylabel('Mean response', fontsize=6)
    ax.flatten()[i].set_title(rec['resp'].chans[i], fontsize=6)

f.tight_layout()

f, ax = plt.subplots(1, 1)

ax.set_title('Population Tuning')
ax.plot(np.stack(ftc_all).mean(axis=0))
ax.set_xticks(range(np.stack(ftc_all).shape[-1]))
ax.set_xticklabels(cfs, rotation=45)
ax.set_xlabel('CF')
ax.set_ylabel('Mean response')

f.tight_layout()

# ================================================================================================
# project responses onto first two PCs across all stims
zscore = True
rall = rec.copy()
rall = rall.create_mask(True)
rall = rall.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT'])
# can't simply extract evoked for refs because can be longer/shorted if it came after target 
# and / or if it was the last stim.So, masking prestim / postim doesn't work.Do it manually
d = rall['resp'].extract_epochs(all_stims) #, mask=rall['mask'])
d = {k: v[~np.isnan(v[:, :, prestim:poststim].sum(axis=(1, 2))), :, :] for (k, v) in d.items()}
d = {k: v[:, :, prestim:poststim] for (k, v) in d.items()}

# zscore each neuron
m = np.concatenate([d[e] for e in d.keys()], axis=0).mean(axis=-1).mean(axis=0)
sd = np.concatenate([d[e] for e in d.keys()], axis=0).mean(axis=-1).std(axis=0)
if zscore:
    d = {k: (v.transpose(0, -1, 1) - m).transpose(0, -1, 1)  for (k, v) in d.items()}
    d = {k: (v.transpose(0, -1, 1) / sd).transpose(0, -1, 1)  for (k, v) in d.items()}

Rall_u = np.vstack([d[k].sum(axis=2).mean(axis=0) for k in d.keys()])

pca = PCA(n_components=2)
pca.fit(Rall_u)
pc_axes = pca.components_

ra = rec.copy().create_mask(True)
ra = ra.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL'])
da = ra['resp'].extract_epochs(all_stims, mask=ra['mask'])
da = {k: v[:, :, prestim:poststim] for (k, v) in da.items()}
da = {k: v[~np.isnan(v.sum(axis=(1, 2))), :, :] for (k, v) in da.items()}
if zscore:
    da = {k: (v.transpose(0, -1, 1) - m).transpose(0, -1, 1)  for (k, v) in da.items()}
    da = {k: (v.transpose(0, -1, 1) / sd).transpose(0, -1, 1)  for (k, v) in da.items()}

rp = rec.copy().create_mask(True)
rp = rp.and_mask(['PASSIVE_EXPERIMENT'])
dp = rp['resp'].extract_epochs(all_stims, mask=rp['mask'])
dp = {k: v[:, :, prestim:poststim] for (k, v) in dp.items()}
dp = {k: v[~np.isnan(v.sum(axis=(1, 2))), :, :] for (k, v) in dp.items()}
if zscore:
    dp = {k: (v.transpose(0, -1, 1) - m).transpose(0, -1, 1)  for (k, v) in dp.items()}
    dp = {k: (v.transpose(0, -1, 1) / sd).transpose(0, -1, 1)  for (k, v) in dp.items()}

# project active / passive responses onto PCA plane
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

for e in all_stims:
    passive = dp[e].mean(axis=-1).dot(pc_axes.T)
    active = da[e].mean(axis=-1).dot(pc_axes.T)

    if e in ref_stims:
        ax[0].plot(passive[:, 0], passive[:, 1], alpha=0.3, marker='.', color='lightgrey', lw=0)
        el = compute_ellipse(passive[:, 0], passive[:, 1])
        ax[0].plot(el[0], el[1], lw=1, color=ax[0].get_lines()[-1].get_color())
        
        ax[1].plot(active[:, 0], active[:, 1], alpha=0.3, marker='.', color='lightgrey', lw=0)
        el = compute_ellipse(active[:, 0], active[:, 1])
        ax[1].plot(el[0], el[1], lw=1, color=ax[1].get_lines()[-1].get_color())
    else:
        ax[0].plot(passive[:, 0], passive[:, 1], alpha=0.3, marker='.', lw=0, label=e)
        el = compute_ellipse(passive[:, 0], passive[:, 1])
        ax[0].plot(el[0], el[1], lw=1, color=ax[0].get_lines()[-1].get_color())
        
        ax[1].plot(active[:, 0], active[:, 1], alpha=0.3, marker='.', lw=0, label=e)
        el = compute_ellipse(active[:, 0], active[:, 1])
        ax[1].plot(el[0], el[1], lw=1, color=ax[1].get_lines()[-1].get_color())

ax[0].axhline(0, linestyle='--', lw=2, color='grey')
ax[0].axvline(0, linestyle='--', lw=2, color='grey')
ax[1].axhline(0, linestyle='--', lw=2, color='grey')
ax[1].axvline(0, linestyle='--', lw=2, color='grey')

ax[0].set_title('Passive')
ax[1].set_title('Active')
ax[0].set_xlabel(r'$PC_1$ (var. explained: {})'.format(round(pca.explained_variance_ratio_[0], 3)))
ax[1].set_xlabel(r'$PC_1$ (var. explained: {})'.format(round(pca.explained_variance_ratio_[0], 3)))
ax[0].set_ylabel(r'$PC_2$ (var. explained: {})'.format(round(pca.explained_variance_ratio_[1], 3)))
ax[1].set_ylabel(r'$PC_2$ (var. explained: {})'.format(round(pca.explained_variance_ratio_[1], 3)))

ax[0].legend(frameon=False, fontsize=6)

fig.tight_layout()

# ===============================================================================================
# get behavior performance for these targets
options.update({'keep_following_incorrect_trial': False})
perf = manager.get_behavior_performance(**options)
di = dict.fromkeys(sounds)
hr = dict.fromkeys(sounds)
for s in sounds:
    stim = s.split('_')[1]
    di[s] = perf['DI'][stim]
    hr[s] = perf['RR'][stim]

ra_epochs = rec.copy()
ra_epochs = ra_epochs.and_mask(['ACTIVE_EXPERIMENT'])
ra_epochs = ra_epochs.apply_mask(reset_epochs=True)
nTotalTrials = (ra_epochs['resp'].epochs.name=='TRIAL').sum()
firstHalf = int(nTotalTrials / 3)
lastHalf = nTotalTrials - firstHalf

# early trials
perf_early = manager.get_behavior_performance(trials=range(firstHalf), **options)
die = dict.fromkeys(sounds)
hre = dict.fromkeys(sounds)
for s in sounds:
    stim = s.split('_')[1]
    try:
        die[s] = perf_early['DI'][stim]
        hre[s] = perf_early['RR'][stim]
    except:
        die[s] = np.nan
        hre[s] = np.nan

# late trials
perf_late = manager.get_behavior_performance(trials=range(lastHalf, nTotalTrials), **options)
dil = dict.fromkeys(sounds)
hrl = dict.fromkeys(sounds)
for s in sounds:
    stim = s.split('_')[1]
    try:
        dil[s] = perf_late['DI'][stim]
        hrl[s] = perf_late['RR'][stim]
    except:
        dil[s] = np.nan
        hrl[s] = np.nan

f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(die.values(), 'o-', label='early trials')
ax.plot(dil.values(), 'o-', label='late trials')
ax.axhline(0.5, linestyle='--', color='grey', lw=2)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(['Catch', '-5dB', '0dB', 'Pure-tone', 'Reminder'])
ax.set_xlabel('Target SNR')
ax.set_ylabel('Behavior Performance (DI)')

ax.legend(frameon=False)

f.tight_layout()

plt.show()

'''

# ================================================================================================

# ================================================================================================
# compute noise correlations in active / passive
from charlieTools.noise_correlations import compute_rsc
ra = rec.copy()
ra = ra.create_mask(True)
ra = ra.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL'])
ra = ra.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
ra = ra['resp'].extract_epochs(sounds, mask=ra['mask'], allow_incomplete=True)
# collapse over evoked period
ra = {k: v.mean(axis=-1, keepdims=True) for (k, v) in ra.items()}
rsc_active = compute_rsc(ra, chans=rec['resp'].chans)

rp = rec.copy()
rp = rp.create_mask(True)
rp = rp.and_mask(['PASSIVE_EXPERIMENT'])
rp = rp.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
rp = rp['resp'].extract_epochs(sounds, mask=rp['mask'], allow_incomplete=True)
# collapse over evoked period
rp = {k: v.mean(axis=-1, keepdims=True) for (k, v) in rp.items()}
rsc_passive = compute_rsc(rp, chans=rec['resp'].chans)

print("NOISE CORRELATIONS:")
print("Active: {0} \nPassive: {1}".format(round(rsc_active['rsc'].mean(), 3), round(rsc_passive['rsc'].mean(), 3)))
print('\n')


# project into TDR space defined on noise axis for catch/targets and discrimination axis between 
# catch and targets
rnoise = rec.copy()
rnoise = rnoise.create_mask(True)
rnoise = rnoise.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT'])
rnoise = rnoise.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
d = rnoise['resp'].extract_epochs(sounds, mask=rnoise['mask'], allow_incomplete=True)
rnoise = np.vstack([d[k].mean(axis=-1, keepdims=True) - d[k].mean(axis=(0, 2), keepdims=True) for k in d.keys()])
'''