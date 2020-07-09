"""
pupil / face vids were mislabeled between 06.16.2020 and 07.02.2020 
pupil vid should be .avi and face vid should be _2.avi

write script to rename files
"""

import os
import datetime

folder = '/auto/data/daq/Cordyceps/training2020/'
temp_file_dir = '/auto/users/hellerc/tmp_files/'

all_files = os.listdir(folder)
video_files = [f for f in all_files if '.avi' in f]
numdays = 17
base = datetime.datetime.today()
date_range = [base - datetime.timedelta(days=x) for x in range(numdays)]
date_range = ['_'.join([str(d.year), '0'+str(d.month), str(d.day)]) if len(str(d.day))==2 else 
                    '_'.join([str(d.year), '0'+str(d.month), '0'+str(d.day)]) for d in date_range]
# keep videos in date range
video_files = [v for v in video_files if v[10:20] in date_range]

vf_2 = [v for v in video_files if ('_2.avi' in v) & (len(v.split('_'))==7)]
vf = [v for v in video_files if v not in vf_2]

if len(vf) != len(vf_2):
    raise ValueError('lengths should match...')

# for each pair of files, swap names
for f1, f2 in zip(vf, vf_2):
    fp1 = folder + f1
    fpt1 = temp_file_dir + f1
    fp2 = folder + f2
    fpt2 = temp_file_dir + f2

    # move f1 to temp file
    os.system('cp {0} {1}'.format(fp1, fpt1))
    # move f2 to temp file (to be safe, have it as a backup)
    os.system('cp {0} {1}'.format(fp2, fpt2))
    
    # copy temp files into their new names in the data dir
    os.system('cp {0} {1}'.format(fpt2, fp1))
    os.system('cp {0} {1}'.format(fpt1, fp2))