"""
test load TNL stimuli from physiology expt
"""
import matplotlib.pyplot as plt
from nems_lbhb.baphy_experiment import BAPHYExperiment

parmfile = '/auto/data/daq/Cordyceps/CRD002/CRD002a10_a_TNL.m'

manager = BAPHYExperiment(parmfile)

# get recording (w/o resp or pupil)
rec = manager.get_recording(**{'pupil': False, 'resp': False, 'rasterfs': 10})

# get all stim epochs
stims = [s for s in rec.epochs.name.unique() if 'STIM_' in s]

# count occurences of each
counts = dict.fromkeys(stims)
for s in stims:
    counts[s] = (rec.epochs.name==s).sum()

plt.plot(counts.values())

plt.show()