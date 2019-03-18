from pathlib import Path
from shutil import copy
import os

if __name__ == '__main__':
    test_dir = '/media/data2/sivankeret/cycle_tibetan_data_long/testB'
    os.makedirs(test_dir, exist_ok=True)
    def copy_test(im):
        path = Path(im)
        new_im_name = str(path.parents[1].name + '%' + path.parents[0].name + '%' + path.name)
        out_path = os.path.join(test_dir, new_im_name)
        copy(im, out_path)
        return 0

    train_dir = '/media/data2/sivankeret/cycle_tibetan_data_long/trainB'
    val_dir = '/media/data2/sivankeret/cycle_tibetan_data_long/valB'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    def copy_val(im):
        copy(im, val_dir)
    def copy_train(im):
        copy(im, train_dir)