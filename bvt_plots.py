import matplotlib.pyplot as plt
import numpy as np
import copy

import charlieTools.plotting as cplt
import charlieTools.preprocessing as preproc
import charlieTools.tuning_curves as tc

def plot_cell_summary(rec1, rec2):
    """
    Plot one page summary of single cell responses for BVT tone discrimination task.
    rec1 is high sampling rate (for raster plotting)
    rec2 is low sampling rate (for other plots, such as psth's)

    Produce a one-page summary of each cell. This includes:
        1) Time courses:
            pupil trace
            target responses (smoothed)
            refs responses nearest tar freqs?
        2) FTC per file
            w/ and w/o pupil regression
        3) PSTH / raster per file
            w/ and w/o pupil regression

    """

    # apply mask to recording and get global params, like tar list
    rec1 = rec1.apply_mask(reset_epochs=True)
    rec2 = rec2.apply_mask(reset_epochs=True)

    targets = [str(t) for t in np.sort([t for t in rec1.epochs.name.unique() if 'TAR_' in t])]

    # layout figure grid (define axes objects)
    files = [f for f in rec1.epochs.name.unique() if 'FILE_' in f]
    nf = len(files)

    fig = plt.figure(figsize=(16, 12))
    pupil_trace = plt.subplot2grid((6, nf), (0, 0), colspan=nf)
    target_resp = plt.subplot2grid((6, nf), (1, 0), colspan=nf)

    ftc_pl = []
    psth = []
    ftc_pl_pr = []
    psth_pr = []
    for i in range(nf):
        ftc_pl.append(plt.subplot2grid((6, nf), (2, i), colspan=1))
        psth.append(plt.subplot2grid((6, nf), (3, i), colspan=1))
        ftc_pl_pr.append(plt.subplot2grid((6, nf), (4, i), colspan=1))
        psth_pr.append(plt.subplot2grid((6, nf), (5, i), colspan=1))

    
    # plot time courses
    time = np.arange(0, rec2['pupil'].shape[-1] / rec2['resp'].fs, 1 / rec2['resp'].fs)
    
    # pupil time course
    pupil_trace.plot(time, rec2['pupil']._data.T, color='green', label='pupil')
    act = rec2['resp'].epoch_to_signal('ACTIVE_EXPERIMENT')._data.squeeze() * rec2['pupil']._data.max()
    pupil_trace.fill_between(time, act, color='lightgrey', alpha=0.5)

    pupil_trace.legend(fontsize=8, frameon=False)
    pupil_trace.set_xlabel('Time (s)')
    pupil_trace.set_ylabel('Size (pixels)')

    # target resp time course
    for t in targets:
        ti, resp = get_resp_timecourse(rec2, str(t))
        resp_roll = rolling_window(resp, 5)
        ti_roll = rolling_window(ti, 5)
        resp = np.mean(resp_roll, axis=-1)
        ti = np.median(ti_roll, axis=-1)
        resp_sem = np.std(resp_roll, axis=-1) / np.sqrt(resp_roll.shape[-1])
        target_resp.plot(ti, resp, label=t)
        target_resp.fill_between(ti, resp-resp_sem, resp+resp_sem, 
                        color=target_resp.get_lines()[-1].get_color(), 
                        alpha=0.3, lw=0)

    lim = target_resp.get_ylim()[-1]
    act = rec2['resp'].epoch_to_signal('ACTIVE_EXPERIMENT')._data.squeeze() * lim
    target_resp.fill_between(time, act, color='lightgrey', alpha=0.5)

    target_resp.legend(fontsize=8, frameon=False)
    target_resp.set_xlabel('Time (s)')
    target_resp.set_ylabel('Spk / bin')

    # pupil regressed recording
    pr = preproc.regress_state(rec2, state_sigs=['pupil'], regress=['pupil'])

    ylim = cplt.get_ylim(rec1, fs=20, epochs=targets)
    tr_max = cplt.get_tr_max(rec1, epochs=targets)
    ftc_ylim = 0
    ftc_pr_ylim = 0
    for i, f in enumerate(files):
        r1 = rec1.copy()
        r1 = r1.and_mask([f]).apply_mask(reset_epochs=True)
        r2 = rec2.copy()
        r2 = r2.and_mask([f]).apply_mask(reset_epochs=True)
        r2_pr = pr.copy()
        r2_pr = r2_pr.and_mask([f]).apply_mask(reset_epochs=True)
        
        # plot ftc
        ftc = tc.get_tuning_curves(r2)
        vals = ftc.loc['r'].to_numpy(dtype=np.float).squeeze()
        sem = ftc.loc['sem'].to_numpy(dtype=np.float).squeeze()
        freqs = ftc.loc['r'].columns
        ftc_pl[i].errorbar(freqs, vals, sem, color='k')
        for tar in targets:
            tar = int(tar.split('_')[-1])
            ftc_pl[i].axvline(tar, color='r', linestyle='--')
        
        if ftc_pl[i].get_ylim()[-1] > ftc_ylim:
            ftc_ylim = ftc_pl[i].get_ylim()[-1]
        
        ftc_pl[i].set_ylabel('Spk / bin')
        ftc_pl[i].set_xlabel("CF")

        # plot psth
        cplt.plot_raster_psth(r1, targets, psth_fs=20, ax=psth[i], ylim=ylim, raster=True, tr_max=tr_max)

        # plot ftc pupil regressed
        ftc = tc.get_tuning_curves(r2_pr)
        vals = ftc.loc['r'].to_numpy(dtype=np.float).squeeze()
        sem = ftc.loc['sem'].to_numpy(dtype=np.float).squeeze()
        freqs = ftc.loc['r'].columns
        ftc_pl_pr[i].errorbar(freqs, vals, sem, color='k')
        for tar in targets:
            tar = int(tar.split('_')[-1])
            ftc_pl_pr[i].axvline(tar, color='r', linestyle='--')

        if ftc_pl_pr[i].get_ylim()[-1] > ftc_pr_ylim:
            ftc_pr_ylim = ftc_pl_pr[i].get_ylim()[-1]

        ftc_pl_pr[i].set_ylabel('Spk / bin')
        ftc_pl_pr[i].set_xlabel("CF")

        # plot psth pupil regressed
        cplt.plot_raster_psth(r2_pr, targets, psth_fs=20, ax=psth_pr[i], ylim=ylim, raster=False)
    
    # set ylim for FTC
    for i in range(len(files)):
        ftc_pl[i].set_ylim((0, ftc_ylim))
        ftc_pl_pr[i].set_ylim((0, ftc_pr_ylim))
    fig.tight_layout()

    return fig


def get_resp_timecourse(rec, epoch, evoked=True, poststim=False):
    '''
    return time points, and mean response, for each epoch
    presentation. (mean resp across specified period)
    '''
    r = copy.deepcopy(rec)
    if evoked:
        r = r.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    elif poststim:
        r = r.and_mask(['PostStimSilence'])

    times = 0.5 * (r.epochs[r.epochs.name==epoch].start + r.epochs[r.epochs.name==epoch].end)

    resp = r['resp'].extract_epoch(epoch)
    m = r['mask'].extract_epoch(epoch)

    resp = resp[:, :, m[0, 0, :]]

    resp = resp.mean(axis=-1).squeeze()

    return times.values, resp


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window +1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)