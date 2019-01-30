from os.path import basename, join, exists
from glob import glob
text_files = '/media/data1/sivankeret/ocr_datasets/Hebrew/Dataset/Orig/lines/Texts/*.txt'
images_dir = '/media/data1/sivankeret/ocr_datasets/Hebrew/Dataset/Orig/lines/cropped'
out_file = '/media/data1/sivankeret/ocr_datasets/Hebrew/Dataset/Orig/lines/dataset.txt'

if exists(out_file):
    raise Exception("file should not exist.")

with open(out_file, 'w') as out_f:
    for txt_file in glob(text_files):
        with open(txt_file, 'r') as txt_f:
            text_lines = txt_f.readlines()
        im_files = glob(join(images_dir, basename(txt_file)[:-4] + '_*.*'))
        im_files = sorted(im_files)
        if len(im_files) != len(text_lines):
            raise Exception("There should same number of images as text lines: {} images, {} texts.".format(len(im_file), len(text_lines)))

        for img, txt in zip(im_files, text_lines):
            out_f.writelines(basename(img) + '   *   ' + txt + '\n')

