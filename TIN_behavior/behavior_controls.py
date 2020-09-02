"""
Plot distribution of catch slots vs. target slots. Are they balanced?
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import os
import numpy as np
import matplotlib.pyplot as plt
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
                                            (runclass in f) & (f.endswith('.m') & (site in f) & ('_a_' in f))]
parmfiles = [f for f in parmfiles if (int(os.path.basename(f)[7:9]) in run_nums)]

manager = BAPHYExperiment(parmfile=parmfiles)
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()

events = manager.get_behavior_events(**options)[0]
targets = [n for n in events.name.unique() if ('Target' in n) & ('StimSilence' not in n)]
targets = (np.array(targets)[np.argsort([int(t.split(',')[1][-2:]) for t in targets])]).tolist()
catch = [n for n in events.name.unique() if ('Catch' in n) & ('StimSilence' not in n)]
trials = [t for t in events.Trial.unique() if events[events.Trial==t]['invalidTrial'].sum()!=sum(events.Trial==t)]

tar_slots = {t: [] for t in targets}
catch_slots = {t: [] for t in catch}
for t in trials:
    evt = events[events.Trial==t]
    tstims = [e for e in evt.name if 'Stim , ' in e]
    for i, s in enumerate(tstims):
        if s in targets:
            tar_slots[s].append(i)
        elif s in catch:
            catch_slots[s].append(i)

f, ax = plt.subplots(1, 1, figsize=(8, 4))
bins = np.arange(1, 9)
tit_str = []
for t in tar_slots.keys():
    ax.hist(tar_slots[t], bins=bins, histtype='step', lw=1, label=t)
    tit_str.append('{0}: {1}\n'.format(t.split(',')[1], round(np.mean(tar_slots[t]), 3)))
for c in catch_slots.keys():
    ax.hist(catch_slots[c], bins=bins, histtype='step', lw=1, label=c)
    tit_str.append('{0}: {1}\n'.format(c.split(',')[1], round(np.mean(catch_slots[c]), 3)))
ax.set_xlabel('Target/Catch slot')
ax.set_ylabel('Occurences')

ax.set_title(''.join(tit_str))

ax.legend(frameon=False, fontsize=6)

f.tight_layout()

plt.show()