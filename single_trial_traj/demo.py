"""
Look at mean single trial trajectories for HIT, MISS, CORRECT_REJECT, PASSIVE etc.

Only look at target window for now, and plot trajectory for each target sep.

Dim reduce using PCA on trial averaged target responses, unwrapped in time over trial
"""
import nems.db as nd
from nems.recording import Recording
import nems_lbhb.baphy as nb
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

import charlieTools.plotting as cplt

site = 'DRX005c.e65:128'
batch = 302
single_PC = False

rasterfs = 20
ops = {'batch': batch, 'cellid': site, 'rasterfs': rasterfs, 'pupil': 1, 'stim': 0, 'recache': False}
uri = nb.baphy_load_recording_uri(**ops)
rec = Recording.load(uri[0])
rec['resp'] = rec['resp'].rasterize()

if shank=='left':
    left_chans = [c for c in rec['resp'].chans if int(c.split('-')[-2])<=64]
    rec['resp'] = rec['resp'].extract_channels(left_chans)
elif shank=='right':
    right_chans = [c for c in rec['resp'].chans if int(c.split('-')[-2])>64]
    rec['resp'] = rec['resp'].extract_channels(right_chans)
else:
    pass

# mask only target responses
targets = [t for t in rec.epochs.name.unique() if 'TAR_' in t]
targets = np.sort(targets).tolist()
rec = rec.and_mask(targets)

# apply mask
rec = rec.apply_mask(reset_epochs=True)

# create trajectory plot for each file
files = [f for f in rec.epochs.name.unique() if 'FILE' in f]
if site == 'DRX005c':
    files.remove('FILE_DRX005c07_p_BVT')

if single_PC:
    # do PCA over all data
    pass

for f in files:

    if ~single_PC:
        # do PCA for this file alone
        _rec = rec.and_mask(f)
        _rec = _rec.apply_mask(reset_epochs=True)
        R = _rec['resp']._data
        pca = PCA(n_components=2)
        pca.fit(R.T)

    # Project mean hit trial, miss trial, incorrect rej. trial, 
    # correct reject trial, passive trial onto PCs

    # HITS (for each target?)
    f, ax = plt.subplots(1, 1)
    rhit = rec.and_mask(['HIT_TRIAL'])
    for t in targets:
        try:
            r = rhit.and_mask(t)
            r = r.apply_mask(reset_epochs=True)
            rf = r['resp'].extract_epoch('TRIAL')

            rfm = rf.mean(axis=0)

            proj = np.matmul(rfm.T, pca.components_.T)
            # plot prestim, post stim, and evoked sep.
            prestim = int(0.1 * r['resp'].fs)
            evoked = int(0.5 * r['resp'].fs)
            ax.plot(proj[:prestim+1, 0], proj[:prestim+1, 1], 'o-')
            ax.plot(proj[prestim:(prestim+evoked)+1, 0], proj[prestim:(prestim+evoked)+1, 1], '-', color=ax.get_lines()[-1].get_color(), label=t)
            ax.plot(proj[(prestim+evoked):, 0], proj[(prestim+evoked):, 1], '-', alpha=0.5, color=ax.get_lines()[-1].get_color())
        except:
            pass

    # MISSES (for each target?)
    rmiss = rec.and_mask(['MISS_TRIAL'])
    for t in targets:
        try:
            r = rmiss.and_mask(t)
            r = r.apply_mask(reset_epochs=True)
            rf = r['resp'].extract_epoch('TRIAL')

            rfm = rf.mean(axis=0)

            proj = np.matmul(rfm.T, pca.components_.T)
            # plot prestim, post stim, and evoked sep.
            prestim = int(0.1 * r['resp'].fs)
            evoked = int(0.5 * r['resp'].fs)
            ax.plot(proj[:prestim+1, 0], proj[:prestim+1, 1], 'o-')
            ax.plot(proj[prestim:(prestim+evoked)+1, 0], proj[prestim:(prestim+evoked)+1, 1], '-', color=ax.get_lines()[-1].get_color(), label=t)
            ax.plot(proj[(prestim+evoked):, 0], proj[(prestim+evoked):, 1], '-', alpha=0.5, color=ax.get_lines()[-1].get_color())
        except:
            pass


    # PASSIVE TRIALS (FOR REW. TARGETS)
    rpass = rec.and_mask(['PASSIVE_EXPERIMENT'])
    for t in targets:
        try:
            r = rpass.and_mask(t)
            r = r.apply_mask(reset_epochs=True)
            rf = r['resp'].extract_epoch('TRIAL')

            rfm = rf.mean(axis=0)

            proj = np.matmul(rfm.T, pca.components_.T)
            # plot prestim, post stim, and evoked sep.
            prestim = int(0.1 * r['resp'].fs)
            evoked = int(0.5 * r['resp'].fs)
            ax.plot(proj[:prestim+1, 0], proj[:prestim+1, 1], 'o-')
            ax.plot(proj[prestim:(prestim+evoked)+1, 0], proj[prestim:(prestim+evoked)+1, 1], '-', color=ax.get_lines()[-1].get_color(), label=t)
            ax.plot(proj[(prestim+evoked):, 0], proj[(prestim+evoked):, 1], '-', alpha=0.5, color=ax.get_lines()[-1].get_color())
        except:
            pass

    # CORRECT REJECTS

    # INCORRECT REJECTS

