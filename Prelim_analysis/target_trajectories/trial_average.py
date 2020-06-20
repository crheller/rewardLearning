from nems_lbhb.baphy_experiment import BAPHYExperiment
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

batch = 302
site = 'DRX005c'
fs = 40

options = {'batch': batch,
           'cellid': site,
           'rasterfs': fs,
           'pupil': True,
           'resp': True,
           'stim':False}


manager = BAPHYExperiment(batch=batch, siteid=site)
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()
ncells = rec['resp'].shape[0]

# mask out all trials except for early trials / FA trials (where there may be nans)
rec = rec.and_mask(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL'], invert=True)

# Get first 2 PCs of the trial-averaged TARGET responses
targets = [e for e in rec.epochs.name.unique() if 'TAR_' in e]
target_resp = []
for t in targets:
    _r = rec['resp'].extract_epoch(t, mask=rec['mask']).mean(axis=0)
    target_resp.append(_r)

target_resp = np.stack(target_resp).transpose([1, 0, 2]).reshape(ncells, -1)
pca = PCA(n_components=2)
pca.fit(target_resp.T)

rec['tar_pc1'] = rec['resp']._modified_copy(np.matmul(rec['resp'].as_continuous().T, pca.components_[[0]].T).T)
rec['tar_pc2'] = rec['resp']._modified_copy(np.matmul(rec['resp'].as_continuous().T, pca.components_[[1]].T).T)

# plot the three target trajectories for active / passive
reca = rec.copy()
reca = reca.and_mask(['ACTIVE_EXPERIMENT'])
correct = reca.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])
incorrect = reca.and_mask(['INCORRECT_HIT_TRIAL', 'MISS_TRIAL'])
recp = rec.copy()
recp = recp.and_mask(['PASSIVE_EXPERIMENT'])

f, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
f2, ax2 = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

for t in targets:
    pc1a = rec['tar_pc1'].extract_epoch(t, mask=correct['mask']).mean(axis=0).squeeze()
    pc2a = rec['tar_pc2'].extract_epoch(t, mask=correct['mask']).mean(axis=0).squeeze()

    pc1i = rec['tar_pc1'].extract_epoch(t, mask=incorrect['mask']).mean(axis=0).squeeze()
    pc2i = rec['tar_pc2'].extract_epoch(t, mask=incorrect['mask']).mean(axis=0).squeeze()

    pc1p = rec['tar_pc1'].extract_epoch(t, mask=recp['mask']).mean(axis=0).squeeze()
    pc2p = rec['tar_pc2'].extract_epoch(t, mask=recp['mask']).mean(axis=0).squeeze()

    ax[0].plot(pc1a, pc2a, lw=2, label=t)

    ax[1].plot(pc1i, pc2i, lw=2, label=t)

    ax[2].plot(pc1p, pc2p, lw=2, label=t)

    ax2[0].plot(pc1a, lw=2, label=t)

    ax2[1].plot(pc1i, lw=2, label=t)

    ax2[2].plot(pc1p, lw=2, label=t)

ax[0].set_title('Correct trials')
ax[1].set_title('Incorrect trials')
ax[2].set_title('Passive trials')
ax[0].set_ylabel('PC2')
ax[1].set_xlabel('PC1')
ax[0].legend(frameon=False, fontsize=8)

ax2[0].set_title('Correct trials')
ax2[1].set_title('Incorrect trials')
ax2[2].set_title('Passive trials')
ax2[0].set_ylabel('PC1')
ax2[1].set_xlabel('Time')
ax2[0].legend(frameon=False, fontsize=8)

f2.tight_layout()

plt.show()