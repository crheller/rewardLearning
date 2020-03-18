import nems_lbhb.io as io

parmfile = '/auto/data/daq/Drechsler/DRX002/DRX002c09_p_BNB.m'

exp = io.BAPHYExperiment(parmfile)

oep_ts = exp.get_trial_starts()
oep_ts = oep_ts - oep_ts[0]
events = io.baphy_parm_read(parmfile)

events = io.baphy_align_time_openephys(events[-1], oep_ts)

prestim = events[events.name.str.contains('PreStimSilence')]['end']- \
                events[events.name.str.contains('PreStimSilence')]['start']   
poststim = events[events.name.str.contains('PostStimSilence')]['end']- \
                events[events.name.str.contains('PostStimSilence')]['start'] 
trial_len = round(events[events.name.str.contains('PostStimSilence')]['end'].iloc[0] - \
                events[events.name.str.contains('PreStimSilence')]['start'].iloc[0] , 2)

reps = len(oep_ts)

data = exp.get_continuous_data(np.arange(1, 10))
data = data - data.mean(axis=0)
fs = 30000
time = np.arange(0, data.shape[-1] / fs, 1/fs)

nreps = events['Trial'].max()
nchans = data.shape[0]

folded_data = np.zeros((int(trial_len*fs), nreps, nchans))

for r in np.arange(0, nreps):
    s = events[(events.Trial==r+1) & events.name.str.contains('PreStimSilence')]['start']
    e = events[(events.Trial==r+1) & events.name.str.contains('PostStimSilence')]['end']
    idx = (time > s.values[0]) & (time <= e.values[0])
    folded_data[:, r, :] = data[:, idx].reshape(nchans, idx.sum()).T
