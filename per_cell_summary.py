"""
Produce a one-page pdf summary of each cell. This includes:
    1) Time courses:
        pupil trace
        target responses (smoothed)
        refs responses nearest tar freqs?
    2) FTC per file
        w/ and w/o pupil regression
    3) PSTH / raster per file
        w/ and w/o pupil regression
"""

import nems.db as nd
from nems.recording import Recording
import nems_lbhb.baphy as nb

import numpy as np
import matplotlib.pyplot as plt
import copy

import bvt_plots

import logging

log = logging.getLogger(__name__)

batch = 302
cells = nd.get_batch_cells(batch)
cellids = [c for c in cells.cellid if 'DRX' in c]
sites = np.unique([c[:7] for c in cellids])

# loop over sites (to speed loading)
for site in sites:
    # load recording(s)
    rasterfs = 100
    ops = {'batch': batch, 'siteid': site, 'rasterfs': rasterfs, 'pupil': 1, 'stim': 0, 'recache': False}
    uri = nb.baphy_load_recording_uri(**ops)
    rec100 = Recording.load(uri)
    rec100['resp'] = rec100['resp'].rasterize()
    rec100 = rec100.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT'])

    rasterfs = 20
    ops = {'batch': batch, 'siteid': site, 'rasterfs': rasterfs, 'pupil': 1, 'stim': 0, 'recache': False}
    uri = nb.baphy_load_recording_uri(**ops)
    rec20 = Recording.load(uri)
    rec20['resp'] = rec20['resp'].rasterize()
    rec20 = rec20.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT'])

    # loop over all cells in site, now that site is loaded
    cells = [c for c in cellids if site in c]
    for cell in cells:
        r100 = copy.deepcopy(rec100)
        r20 = copy.deepcopy(rec20)

        r100['resp'] = r100['resp'].extract_channels([cell])
        r20['resp'] = r20['resp'].extract_channels([cell])

        f = bvt_plots.plot_cell_summary(r100, r20)

        log.info("save plot for cell {}".format(cell))
        f.savefig('/auto/users/hellerc/code/projects/rewardLearning/singleCellFigs/{}.png'.format(cell))
        plt.close('all')
        