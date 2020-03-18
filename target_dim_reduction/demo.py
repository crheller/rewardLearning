"""
Reduce pop. response to a single dimension (decoding axis, for example) and plot the 
target responses over time on this axis.
"""

import nems.db as nd
from nems.recording import Recording
import nems_lbhb.baphy as nb
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

import charlieTools.plotting as cplt

batch = 302
shank = 'right' # left, right
sites = ['DRX005c', 'DRX006b', 'DRX007a', 'DRX008b']
site = 'DRX005c'
for site in sites:
    # load recoding, mask correct trials
    rasterfs = 20
    ops = {'batch': batch, 'siteid': site, 'rasterfs': rasterfs, 'pupil': 1, 'stim': 0, 'recache': False}
    uri = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
    rec['resp'] = rec['resp'].rasterize()
    if shank=='left':
        left_chans = [c for c in rec['resp'].chans if int(c.split('-')[-2])<=64]
        rec['resp'] = rec['resp'].extract_channels(left_chans)
    elif shank=='right':
        right_chans = [c for c in rec['resp'].chans if int(c.split('-')[-2])>64]
        rec['resp'] = rec['resp'].extract_channels(right_chans)
    else:
        pass
    rec = rec.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT'])
    #rec = rec.and_mask(['HIT_TRIAL', 'INCORRECT_HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT'])

    # mask only target responses
    targets = [t for t in rec.epochs.name.unique() if 'TAR_' in t]
    targets = np.sort(targets).tolist()
    rec = rec.and_mask(targets)

    # apply mask
    rec = rec.apply_mask(reset_epochs=True)

    # create resp / state matrices
    # collapse first 200 ms of target response for simplicity
    nbins = int(rec['resp'].fs * 0.2)
    R = rec['resp'].extract_epoch('TARGET')[:, :, :nbins].mean(axis=-1)
    S = rec['resp'].epoch_to_signal('ACTIVE_EXPERIMENT').extract_epoch('TARGET')[:, :, :nbins].mean(axis=-1)
    tar_masks = {}
    f_masks = {}
    for t in targets:
        tar_masks[t] = rec['resp'].epoch_to_signal(t).extract_epoch('TARGET')[:, :, 0]

    files = [f for f in rec.epochs.name.unique() if 'FILE' in f]
    if site == 'DRX005c':
        files.remove('FILE_DRX005c07_p_BVT')
    for f in files:
        f_masks[f] = rec['resp'].epoch_to_signal(f).extract_epoch('TARGET')[:, :, 0]


    # perform simple PCA to get sense of dimensionality, both in a/p and all together
    fig, ax = plt.subplots(1, 1)
    for f in files:
        pca = PCA()
        _R = np.zeros((len(targets), R.shape[-1]))
        for i, t in enumerate(targets):
            if f_masks[f][:, 0].sum() > 0:
                _R[i, :] = R[f_masks[f][:, 0] & tar_masks[t][:, 0], :].mean(axis=0)
            else: 
                pass
        pca.fit(_R)
        ax.plot(np.cumsum(pca.explained_variance_ratio_), '-o', label=f)

    ax.legend()
    fig.canvas.set_window_title(site)

    # project each target into PC space for each behavior block with pcs calculated 
    # independenlty for each block on the trial averaged data
    fig, ax = plt.subplots(1, len(files), figsize=(12, 6), sharex=True, sharey=True)
    for j, f in enumerate(files):
        pca = PCA()
        _R = np.zeros((len(targets), R.shape[-1]))
        for i, t in enumerate(targets):
            _R[i, :] = R[f_masks[f][:, 0] & tar_masks[t][:, 0], :].mean(axis=0)
        pca.fit(_R)

        # now, project each target single trials onto the first two PCs
        for i, t in enumerate(targets):
            tar_resp = R[f_masks[f][:, 0] & tar_masks[t][:, 0], :]
            proj = np.matmul(tar_resp, pca.components_[:2, :].T)
            ax[j].plot(proj[:, 0], proj[:, 1], '.', label=t)
            el = cplt.compute_ellipse(proj[:, 0], proj[:, 1])
            ax[j].plot(el[0, :], el[1, :], color=ax[j].get_lines()[-1].get_color())
        ax[j].legend(frameon=False, fontsize=8)
        ax[j].set_xlabel('PC1', fontsize=8)
        ax[j].set_ylabel('PC2', fontsize=8)
        ax[j].set_title(f)
        #ax[j].set_aspect(cplt.get_square_asp(ax[j]))
        ax[j].set_aspect('equal')
    fig.tight_layout()
    fig.canvas.set_window_title(site)


    # for each file, compute 1st PC independently, then project all single trials, over time, onto the
    # single axis
    fig, ax = plt.subplots(len(files), 1)
    for j, f in enumerate(files):
        pca = PCA()
        _R = np.zeros((len(targets), R.shape[-1]))
        for i, t in enumerate(targets):
            _R[i, :] = R[f_masks[f][:, 0] & tar_masks[t][:, 0], :].mean(axis=0)
        pca.fit(_R)

        transform = np.matmul(R, pca.components_[0, :][:, np.newaxis])

        m = 0
        mi = 0
        for t in targets:
            ax[j].plot(np.argwhere(tar_masks[t][:, 0]), transform[tar_masks[t][:, 0], 0], 'o', label=t, markersize=5)
            if max(transform[tar_masks[t][:, 0], 0]) > m:
                m = max(transform[tar_masks[t][:, 0], 0])
            if min(transform[tar_masks[t][:, 0], 0]) < mi:
                mi = min(transform[tar_masks[t][:, 0], 0])
        
        af = [f for f in files if '_a_' in f]
        act = f_masks[af[0]].copy()
        for _af in af:
            act += f_masks[_af]
        ab = act.copy()
        act = act.astype(np.float)
        act[ab] = 0
        act[~ab] = np.nan
        ax[j].plot(range(act.shape[0]), act.squeeze(), color='red', lw=3, label='active')

        ax[j].legend(frameon=False, fontsize=8)
        ax[j].set_ylabel('PC1', fontsize=8)
        ax[j].set_title('PCA on: {}'.format(f), fontsize=8)


    #fig.tight_layout()
    fig.suptitle('ncells: {0} \n \
        shank = {1}'.format(len(rec['resp'].chans), shank), fontsize=8)

    fig.canvas.set_window_title(site)
plt.show()
