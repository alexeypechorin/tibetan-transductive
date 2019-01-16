from glob import glob
from random import shuffle
from copy import copy_test, copy_val, copy_train
import os
from os.path import join, basename
import multiprocessing

pool = multiprocessing.Pool(processes=8)
orig_files = '/home/sivankeret/tibetan_synth_data_06/synth2/synth/Images/**/*.png'
orig_txt = ['/home/sivankeret/tibetan_synth_data_06/synth2/synth/data_train.txt',
'/home/sivankeret/tibetan_synth_data_06/synth2/synth/data_val.txt']
out_dir = ['/media/data2/sivankeret/cycle_tybetan_res/only_one/64/tibetan_cycle_identity_05/test_10000',
           '/media/data2/sivankeret/cycle_tybetan_res/only_one/64/tibetan_cycle_identity_05/test_20000'
           ]




all_orig = glob(orig_files,recursive=True)
shuffle(all_orig)
#images_to_val = all_orig[:len(all_orig)//10]
#images_to_train = all_orig[len(all_orig)//10:]
#pool.map(copy_val, images_to_val)
#pool.map(copy_train, images_to_train)
pool.map(copy_test, all_orig)

'''
orig_files = '/home/sivankeret/tibetan_synth_data_06/synth2/synth/Images/**/*.png'
orig_txt = ['/home/sivankeret/tibetan_synth_data_06/synth2/synth/data_train.txt',
'/home/sivankeret/tibetan_synth_data_06/synth2/synth/data_val.txt']
out_dir = ['/media/data2/sivankeret/cycle_tybetan_res/only_one/64/tibetan_cycle_identity_05/test_10000',
           '/media/data2/sivankeret/cycle_tybetan_res/only_one/64/tibetan_cycle_identity_05/test_20000'
           ]
test_dir = '/media/data2/sivankeret/cycle_tibetan/testB'
if __name__ == '__main__':
    for f_path in orig_txt:
        with open(f_path,'r') as f:
            all_lines = f.readlines()
            all_lines = [line.replace('/','%') for line in all_lines]
        for out_p in out_dir:
            out_path = join(out_p,basename(f_path))
            with open(out_path,'w') as f:
                f.writelines(all_lines)
    """
    os.makedirs(test_dir, exist_ok=True)
    all_orig = glob(orig_files,recursive=True)
    shuffle(all_orig)
    images_to_val = all_orig[:len(all_orig)//10]
    images_to_train = all_orig[len(all_orig)//10:]
    pool.map(copy_test, all_orig)
    """
'''