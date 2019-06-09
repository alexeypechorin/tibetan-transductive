import glob, os
import shutil

from line_segmentation import im2lines
import pathlib
from skimage.io import  imsave
from multiprocessing import Pool
import traceback
import tqdm
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

IMAGENAME_LINENUM_SEPARATOR = '_IMLINESEP_'

def images_to_lines(text_dir, im_dir):
    images_paths = glob.glob(os.path.join(im_dir, '*.png'))
    existing_images = [Path(path).name.split(IMAGENAME_LINENUM_SEPARATOR)[0].strip() for path in images_paths]  # orig
    text_paths = glob.glob(os.path.join(text_dir, '*.txt'), recursive=True)
    existing_text = [Path(path).name.split('.')[0].strip() for path in text_paths]
    # valid_data = set([int(d) for d in existing_images]) & set([int(d) for d in existing_text])  # do_intersection
    valid_data = set([str(d) for d in existing_images]) & set([str(d) for d in existing_text])  # do_intersection
    im_to_text = {}
    for data in valid_data:
        cur_pattern = os.path.join(im_dir, '{}_*.png'.format(data))  # [ +]{}_*.png
        cur_images = glob.glob(cur_pattern, recursive=True)
        cur_text_path = os.path.join(text_dir, '{}.txt'.format(data))
        with open(cur_text_path, 'r') as f:
            all_text_lines = f.readlines()
        if len(cur_images) != len(all_text_lines):
            print('Error in image {} - num text lines: {} different from num image lines: {}'.format(
                data, len(all_text_lines), len(cur_images)
            ))
            continue
        for im in cur_images:
            id = int(Path(im).name.split(IMAGENAME_LINENUM_SEPARATOR)[1].split('.')[0])
            line = all_text_lines[id]  # orig
            # line = all_text_lines[id - 1] # tmp
            if len(line) > 1:
                if (not '{' in line) and (not '{' in line):
                    im_to_text[im] = line

    output_lines = [Path(im).name + '   *   ' + line for im, line in im_to_text.items()]
    return output_lines


def proccess_image(im_path, output_dir, verbose=True,
                   tmp_workplace="/home/wolf/alexeyp/ocr_datasets/wiener_transcribed/wiener_zahi_clean/data4debug/tmp_workplace_closing_square15_open_4_20",
                   dataset_name="tibetan"):
    try:
        im_path_obj = pathlib.Path(im_path)
        base_name = extract_base_image_name(im_path_obj, dataset_name)
        line_images = im2lines(im_path, verbose=verbose, tmp_workplace=tmp_workplace, dataset_name=dataset_name)
        for line, image in line_images.items():
            line_path = pathlib.Path(output_dir) / (base_name + IMAGENAME_LINENUM_SEPARATOR + '{}.png'.format(line))
            imsave(str(line_path), image)
    except Exception as e:
        print("Error proccessing file: {}".format(im_path))
        traceback.print_exc()


def extract_base_image_name(im_path_obj, dataset_name):
    base_file_name = im_path_obj.name.split('.')[0]
    if dataset_name == "tibetan":
        return base_file_name.split('-')[1].strip()
    elif dataset_name == "wiener":
        return base_file_name #.split('_')[0].strip()
    else:
        return base_file_name

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_im_dir', type=str,
                        help='folder containing images for line segmentation',
                        default='../../Data/Test/Original/Images')
    parser.add_argument('-t', '--input_text_dir', type=str,
                        help='folder containing texts for line segmentation',
                        default='../../Data/Test/Original/Texts')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='path to folder containing output (should not exist)',
                        default='../../Data/Test/Prepared2')
    parser.add_argument('-v', '--verbose', type=bool,
                        help='output verbose debug data',
                        default=False)
    parser.add_argument('-a', '--tmp_workplace', type=str,
                        help='path to folder for temporary workplace',
                        default='/home/wolf/alexeyp/ocr_datasets/wiener_transcribed/wiener_zahi_clean/data4debug/tmp_workplace_closing_square15_open_4_20')
    parser.add_argument('-d', '--dataset_name', type=str,
                        help='Currently wiener or tibetan',
                        default='tibetan')
    parser.add_argument('-n', '--num_parallel', type=int,
                        help='number of parallel threads to run', default=8)
    parser.add_argument('-b', '--debug', type=bool,
                        help='debug mode - delete output folder, create temporary folder',
                        default=False)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        if args.debug:
            shutil.rmtree(args.output_dir)
        else:
            raise AssertionError('Output folder should not exist: {}'.format(args.output_dir))
    if not os.path.exists(args.tmp_workplace):
        if args.debug:
            os.makedirs(args.tmp_workplace)

    os.makedirs(args.output_dir)
    out_im_dir = os.path.join(args.output_dir, 'LineImages')
    os.makedirs(out_im_dir)
    images_paths = glob.glob(os.path.join(args.input_im_dir, '*.jpg'), recursive=True)
    # split images to lines and save each line image
    proccess_image_partial = partial(proccess_image,
                                     output_dir=out_im_dir, verbose=args.verbose, tmp_workplace=args.tmp_workplace,
                                     dataset_name=args.dataset_name.lower())
    with Pool(5) as p:
        list(tqdm.tqdm(p.imap(proccess_image_partial, images_paths), total=len(images_paths)))
    # do image to line
    output_lines = images_to_lines(args.input_text_dir, out_im_dir)

    output_path = os.path.join(args.output_dir, 'im2line.txt')
    with open(output_path, 'w') as f:
        f.writelines(output_lines)

