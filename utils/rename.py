from glob import glob
from random import shuffle
from shutil import move
import os
import multiprocessing
pool = multiprocessing.Pool(processes=4)
orig_files = '/media/data2/sivankeret/03_tibetan/prepared_alignments/Betsu-Group/line_images_cycle/**/*.png'


for im in glob(orig_files):
    im_no_fake = im.replace('_fake_B','')
    move(im, im_no_fake)