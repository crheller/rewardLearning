import os
from nems_lbhb.pup_py.batch_process import queue_pupil_jobs, mark_complete

pupil_python = '/auto/users/hellerc/anaconda3/envs/pupil_analysis/bin/python'

path = '/auto/data/daq/Cordyceps/'
pens = ['CRD002', 'CRD003', 'CRD004', 'CRD008', 'CRD009', 'CRD010', 'CRD011', 'CRD012', 'CRD013', 'CRD015', 'CRD016', 'CRD017']
runclass = ['TBP', 'NAT']

videos = []
for pen in pens:
    files = os.listdir(os.path.join(path, pen))
    files = [f for f in files if f.endswith('.avi') & ~f.endswith('_2.avi')]
    files = [f for f in files if os.path.splitext(f)[0].split('_')[-1] in runclass]

    videos.extend(files)

queue_vids = []
for v in videos:
    tf = input("Do you want to queue the following pupil video: {0}? (Y/n)    ".format(v))

    if tf in ['Y', 'y']:
        queue_vids.append(v)
    elif tf in ['N', 'n']:
        pass
    else:
        raise ValueError("respond to the question with 'N'/'n' or 'Y'/'y'!")

queue_pupil_jobs(queue_vids, python_path=pupil_python, username='hellerc', force_rerun=True)

print("Added jobs to queue. When finished running, come back here to 'batch' mark as complete" \
        "\n AFTER PERFORMING QC check with pupil_browser!")

if 0:
    mark_complete(queue_vids)
