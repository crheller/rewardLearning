"""
Plot population target response matrices sorted by trial outcome.
z-score spikes to better see responses of each neuron.

Categories:
    HIT_TRIAL (rew. target)
    MISS_TRIAL (rew. target)
    CORRECT_REJECT_TRIAL (n. rew. target)
    INCORRECT_HIT_TRIAL  (n. rew. target)
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import charlieTools.tuning_curves as tc
import charlieTools.reward_learning.behavior_helpers as bh

batch = 302
site = 'DRX008b'  # DRX005c, DRX006b, DRX007a, DRX008b
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

# only include TARGET response data
rec = rec.and_mask(['TARGET'])
rec = rec.apply_mask(reset_epochs=True)

# z-score response of all neurons
r = rec['resp']._data 
rz = r - r.mean(axis=-1, keepdims=True)
rz = rz / rz.std(axis=-1, keepdims=True)
rec['resp'] = rec['resp']._modified_copy(rz)

# Pull out the four response matrices for each behavior outcome

# HIT
hit = rec['resp'].extract_epoch('HIT_TRIAL')
# MISS
try:
    miss = rec['resp'].extract_epoch('MISS_TRIAL')
except:
    miss = None
# CORR. REJECT
cmiss = rec['resp'].extract_epoch('CORRECT_REJECT_TRIAL')
# INCORR. HIT
ihit = rec['resp'].extract_epoch('INCORRECT_HIT_TRIAL')

# sort units based on the first PC of the correct trials
pca = PCA(n_components=1)
idx = np.argsort(pca.fit(hit.mean(axis=0).T).components_[0])
idx2 = np.argsort(pca.fit(cmiss.mean(axis=0).T).components_[0])

vmin = -3
vmax = 3
f, ax = plt.subplots(2, 2, figsize=(6, 8))

ax[0, 0].imshow(hit[:, idx, :].mean(axis=0), aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
ax[0, 0].set_title('Hit')

if miss is not None:
    ax[0, 1].imshow(miss[:, idx, :].mean(axis=0), aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
ax[0, 1].set_title('Miss')

ax[1, 0].imshow(cmiss[:, idx2, :].mean(axis=0), aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
ax[1, 0].set_title('Corr. Rej.')

ax[1, 1].imshow(ihit[:, idx2, :].mean(axis=0), aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
ax[1, 1].set_title('Incorr. Hit')

f.tight_layout()

plt.show()