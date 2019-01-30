import glob, os
from image_line_segmentation import im2lines
import pathlib
from skimage.io import  imread, imsave
from multiprocessing import Pool
import traceback
import tqdm
import argparse
from skimage.transform import resize
from functools import partial
from skimage import color

root_dir = '/home/wolf/alexeyp/'

def read_resize_image(img_path, max_width, max_hight):
    img = imread(img_path)
    print(img.shape)
    w,h,c = img.shape
    if c == 4:
        img = color.rgba2rgb(img)
    if w > max_width or h > max_hight:
        w = min(max_hight, (h/max_hight)*w)
        h = min(max_hight, (w/max_width)*h)
        img = resize(img, (w,h))
    return img


def proccess_image(im_path, max_width, max_hight):
    try:

        im_path_obj = pathlib.Path(im_path)
        base_im_dir = im_path_obj.name.replace(" ", '')[:-4]
        tmp_workplace = im_path_obj.parents[1] / 'line_images_color_workplace'/ im_path_obj.parents[1].name/ base_im_dir
        base_path = im_path_obj.parents[1] / 'line_images_color'/ im_path_obj.parents[1].name
        tmp_workplace.mkdir(parents=True, exist_ok=True)
        base_path.mkdir(parents=True, exist_ok=True)
        base_name = im_path_obj.name.split('.')[0].split('-')[1]
        read_resize_image(im_path, max_width, max_hight)
        line_images = im2lines(im_path, tmp_workplace=tmp_workplace, verbose=True, addaptive=False)
        for line, image in line_images.items():
            line_path = base_path / (base_name + '_{}.png'.format(line))
            imsave(str(line_path), image)
    except Exception as e:
        print("Error proccessing file: {}".format(im_path))
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files_pattern',
                        type=str, default=root_dir + 'ocr_datasets/Hebrew/Dataset/Orig/Images/*.png',
                        help=('tibetan_dir'))
    parser.add_argument('--do_parallel', default=False, action='store_true')
    parser.add_argument('--max_width', default=1920)
    parser.add_argument('--max_hight', default=1080)
    args = parser.parse_args()

    #images_pattern = '/media/data2/sivankeret/TibetanMount/prepared_alignments/Betsu-Group/[Ii]mages/39-4/[Ii]mages/*'
    images_pattern = args.input_files_pattern
    images_paths = glob.glob(images_pattern, recursive=True)
    partial_proccess_image = partial(proccess_image, max_width=args.max_width, max_hight=args.max_hight)
    # proccess_image(images_paths[0])
    if args.do_parallel:
        with Pool(5) as p:
            list(tqdm.tqdm(p.imap(partial_proccess_image, images_paths), total=len(images_paths)))
    else:
        for image_path in tqdm.tqdm(images_paths):
            partial_proccess_image(image_path)

