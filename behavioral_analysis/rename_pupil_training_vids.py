"""
face vids (only) were collected between 05/27 and 06/15.
These should be named with _2 to avoid confusion with pupil videos

write script to rename files
"""

import os
import datetime at dt

folder = '/auto/data/daq/Cordyceps/training2020/'
temp_file_dir = '/auto/users/hellerc/tmp_files/'

all_files = os.listdir(folder)
video_files = [f for f in all_files if '.avi' in f]
ed = '2020_05_27'
ld = '2020_06_15'
ed = dt.datetime.strptime(ed, '%Y_%m_%d')
ld = dt.datetime.strptime(ld, '%Y_%m_%d')

vid_dates = [dt.datetime.strptime('-'.join(x.split('_')[1:4]), '%Y-%m-%d') for x in video_files]

# keep videos in date range
video_files = [v for i, v in zip(vid_dates, video_files) if (i >= ed) & (i <= ld)]

for v in video_files:
    v1 = os.path.join(folder, v)
    v2 = os.path.join(folder, v.replace('.avi', '_2.avi'))
    os.system('mv {0} {1}'.format(v1, v2))    