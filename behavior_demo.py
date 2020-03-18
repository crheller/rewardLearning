import matplotlib.pyplot as plt
import numpy as np

from nems_lbhb.baphy_experiment import BAPHYExperiment

parmfile = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_02_14_BVT_3.m'

options = {'pupil': True, 'rasterfs': 100}  # letting beahvior options get set to defaults

manager = BAPHYExperiment(parmfile=parmfile)

# load recording, if you want it...
'''
rec = manager.get_recording(**options)

# example of how to find valid trial numbers from the recording
totalTrials = rec['pupil'].extract_epoch('TRIAL').shape[0]
rec = rec.and_mask('INVALID_BAPHY_TRIAL', invert=True)
validTrials = rec['pupil'].extract_epoch('TRIAL', mask=rec['mask']).shape[0]

# find valid trial numbers using rec epochs times
invalid_time = rec['pupil'].get_epoch_bounds('INVALID_BAPHY_TRIAL') 
trial_epochs = rec['pupil'].get_epoch_bounds('TRIAL') 
good_trials = [i+1 for i in range(0, trial_epochs.shape[0]) if trial_epochs[i] not in invalid_time]
'''

# RT histogram
trials = None # use all baphy trials, or set to specific range
performance = manager.get_behavior_performance(trials=trials, **options)

