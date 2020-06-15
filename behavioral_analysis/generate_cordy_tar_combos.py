"""
Generate list of target frequency / loudness combos for behavioral training.

This is based on results from Cordy 11_08_2019 -> 06_12_2020 where we found that
he seems to be biased towards midrange freqs. (see rewardLearningFigs/)

Change relative loudness to compensate for bias
500 Hz - 2000 Hz : 0 dB
2000 Hz - 5000 Hz : -10 dB
5000 Hz - 16000 Hz : -5 dB

# Generate freq. pairs that are at least 1 octave apart
"""

import numpy as np
import matplotlib.pyplot as plt

# generate list of targets that span the freq range we use for bandpass noise
# (500 Hz - 16000 Hz, 15 steps, logspace)
