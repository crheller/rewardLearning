"""
Plot mean target trajectories for each behavioral outcome category
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import charlieTools.tuning_curves as tc
import charlieTools.reward_learning.behavior_helpers as bh

batch = 302
site = 'DRX007a'  # DRX005c, DRX006b, DRX007a, DRX008b
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


# define low-D projection
all_resp = np.concatenate((hit.mean(axis=0),
                           cmiss.mean(axis=0),
                           ihit.mean(axis=0)), axis=-1)
if miss is not None:
    all_resp = np.concatenate((all_resp, miss.mean(axis=0)), axis=-1)

pca = PCA()
pca.fit(all_resp.T)

f, ax = plt.subplots(1, 4, figsize=(12, 3))
for r, lab in zip([hit, miss, cmiss, ihit], ['HIT', 'MISS', 'Cor. Rej', 'Incor. Hit']):
    if r is not None:
        trajs = r.transpose([0,2,1]).dot(pca.components_.T)
        trajs = sf.gaussian_filter1d(trajs, axis=1, sigma=1.2)
        ax[0].plot(trajs.mean(axis=0)[:,0], trajs.mean(axis=0)[:,1], label=lab)

        ax[1].plot(trajs.mean(axis=0)[:,0])

        ax[2].plot(trajs.mean(axis=0)[:,1])

        ax[3].plot(trajs.mean(axis=0)[:,2])

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_title('PC1')
ax[2].set_title('PC2')    
ax[3].set_title('PC3')  
ax[0].legend(frameon=False)
f.tight_layout()
plt.show()