import matplotlib.pyplot as plt
import numpy as np

from nems_lbhb.baphy_experiment import BAPHYExperiment

parmfile = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_02_14_BVT_3.m'

options = {'pupil': True, 'rasterfs': 100}

manager = BAPHYExperiment(parmfile=parmfile)
rec = manager.get_recording(**options)
rec = rec.and_mask('INVALID_BAPHY_TRIAL', invert=True)
rec = rec.apply_mask(reset_epochs=True)

# plot CORRECT_REJECT target responses vs. INCORRECT_HITS vs. HITS vs. MISSES
crm = rec.and_mask('CORRECT_REJECT_TRIAL')['mask']
icm = rec.and_mask('INCORRECT_HIT_TRIAL')['mask']
hm = rec.and_mask('HIT_TRIAL')['mask']
mm = rec.and_mask('MISS_TRIAL')['mask']

f, ax = plt.subplots(1, 1)

cor_rej = rec['pupil'].extract_epoch('TARGET', mask=crm)
incor_hit = rec['pupil'].extract_epoch('TARGET', mask=icm)
hit = rec['pupil'].extract_epoch('TARGET', mask=hm)
miss = rec['pupil'].extract_epoch('TARGET', mask=mm)

t = np.arange(0, cor_rej.shape[-1] / options['rasterfs'], 1 / options['rasterfs'])
m = cor_rej.mean(axis=0).squeeze()
sem = cor_rej.std(axis=0).squeeze() / np.sqrt(cor_rej.shape[0])
ax.plot(t, m, color='blue', label='CORRECT_REJECT')
ax.fill_between(t, m-sem, m+sem, color='blue', alpha=0.4, lw=0)

t = np.arange(0, incor_hit.shape[-1] / options['rasterfs'], 1 / options['rasterfs'])
m = incor_hit.mean(axis=0).squeeze()
sem = incor_hit.std(axis=0).squeeze() / np.sqrt(incor_hit.shape[0])
ax.plot(t, m, color='coral', label='INCORRECT_HIT')
ax.fill_between(t, m-sem, m+sem, color='coral', alpha=0.4, lw=0)

t = np.arange(0, hit.shape[-1] / options['rasterfs'], 1 / options['rasterfs'])
m = hit.mean(axis=0).squeeze()
sem = hit.std(axis=0).squeeze() / np.sqrt(hit.shape[0])
ax.plot(t, m, color='red', label='HIT')
ax.fill_between(t, m-sem, m+sem, color='red', alpha=0.4, lw=0)

t = np.arange(0, miss.shape[-1] / options['rasterfs'], 1 / options['rasterfs'])
m = miss.mean(axis=0).squeeze()
sem = miss.std(axis=0).squeeze() / np.sqrt(miss.shape[0])
ax.plot(t, m, color='lightblue', label='MISS')
ax.fill_between(t, m-sem, m+sem, color='lightblue', alpha=0.4, lw=0)

ax.set_ylabel('pupil size')
ax.set_xlabel('Time')
ax.set_title('Target pupil response')
ax.legend()

plt.show()
