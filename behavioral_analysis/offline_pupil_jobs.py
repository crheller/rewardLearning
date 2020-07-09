import os
import datetime as dt
from nems_lbhb.pup_py.batch_process import queue_pupil_jobs, mark_complete

folder = '/auto/data/daq/Cordyceps/training2020/'
temp_file_dir = '/auto/users/hellerc/tmp_files/'
pupil_python = '/auto/users/hellerc/anaconda3/envs/pupil_analysis/bin/python'

all_files = os.listdir(folder)
video_files = [f for f in all_files if '.avi' in f]
ed = '2020_06_16'
ld = '2020_06_29'
ed = dt.datetime.strptime(ed, '%Y_%m_%d')
ld = dt.datetime.strptime(ld, '%Y_%m_%d')

vid_dates = [dt.datetime.strptime('-'.join(x.split('_')[1:4]), '%Y-%m-%d') for x in video_files]

# keep videos in date range
video_files = [v for i, v in zip(vid_dates, video_files) if (i >= ed) & (i <= ld)]
video_files = [os.path.join(folder, v) for v in video_files]

# remove face videos
video_files = video_files[0::2][:-1]

# ====================== QUEUE JOBS ========================
queue_pupil_jobs(video_files, python_path=pupil_python, username='hellerc', force_rerun=False)

# ========================= SAVE RESULTS ======================
# update complete status (save predictions and update celldb)
# make sure jobs ^^ are done running first and that you've performed
# sufficient quality control using pupil_browser
if 0:
    mark_complete(video_files)