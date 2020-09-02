import os
from nems_lbhb.pup_py.batch_process import queue_pupil_jobs, mark_complete

pupil_python = '/auto/users/hellerc/anaconda3/envs/pupil_analysis/bin/python'

pen = 'CRD010'
animal = 'Cordyceps'

path = '/auto/data/daq/{0}/{1}/'.format(animal, pen)
video_files = [os.path.join(path, v) for v in os.listdir(path) if ('TBP' in v) & ('.avi' in v) & ('2.avi' not in v)]

queue_pupil_jobs(video_files, python_path=pupil_python, username='hellerc', force_rerun=True)
