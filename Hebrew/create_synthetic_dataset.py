# -*- coding: utf-8 -*-
import glob
import os
import pathlib
import pickle as pkl
import re
import subprocess
from concurrent.futures.process import ProcessPoolExecutor as Executor
import numpy as np
from image_line_segmentation import im2lines
import traceback
from multiprocessing import Pool, Lock
import tqdm
from functools import partial
from numpy.random import choice
from skimage import io
import argparse
from itertools import islice
from random import randint, shuffle
import time

root_dir = '/home/wolf/alexeyp'
god_replace = '****'

def clean_hebrew(text, god_list):
    for word in god_list:
        text = text.replace(' ' + word + ' ', ' ' + god_replace + ' ')
    text = text.replace("ר'", "רבי")
    text = text.replace(' א"ר ', " ")
    text = text.replace('\n', " ")
    re.sub(r'\([^)]*\)', '', text)
    remove_letters = ['(', ')', "'", '"', '[', ']', "<", ">", ".", ",", "`", "-"]
    for letter in remove_letters:
        text = text.replace(letter, " ")
    return text

def create_god_name(texts_list, god_fonts):
    new_god = "<span font_family='{}'>יהוה</span>"
    texts_list = [new_god.format(god_fonts[choice(len(god_fonts))]) if god_replace in text else text for text in texts_list]
    '''
    for god_name in god_list:
        text.replace(god_name, new_god)
    all_matches = re.finditer("{}", text)
    shuffle(all_matches)
    font_size = len(all_matches) // len(god_fonts)
    for i, font in enumerate(god_fonts):
        all_matches[i*font_size,(i+1)*font_size]
        for match in all_matches:
            text = text[:match.start()] + font + text[match.end():]
    '''
    return texts_list


def batch(iterable, bs=1):
    new_i = 0
    len_it = len(iterable)
    while new_i < len_it-1:
        old_i = new_i
        new_i += bs
        yield iterable[old_i:min(old_i + bs, len_it)]


def split_to_lines(li, data_info_list, data_info_probs):
    data_info_i = data_info_list[choice(len(data_info_list), 1, p=data_info_probs)[0]]
    word_list = list(batch(li,randint(data_info_i.min_len_word, data_info_i.min_len_word)))
    txt = [SynthDataInfo.word_seperator.join(word) for word in word_list]
    im_txt = [data_info_i.spaces[choice(len(data_info_i.spaces))] + data_info_i.spaces[choice(len(data_info_i.spaces))].join(
        create_god_name(word, data_info_list[0].god_fonts_names)) for word in word_list]
    return txt, im_txt

class SynthDataInfo:
    word_seperator = ' '
    def __init__(self, multi_line, font_names, god_fonts_names):
        self.multi_line = multi_line
        self.font_names = font_names
        self.god_fonts_names = god_fonts_names
        self.spaces = [' ', '  ', '   ']
        self.min_spaces = 1
        self.max_spaces = 3
        self.min_len_word = 8
        self.max_len_word = 25

'''
should be: text, min_len = 130, max_len = 180, min_allowed_len = 80, max_allowed_len = 200
'''
def create_line_texts(text, data_info_list, data_info_probs, god_list):
    text = clean_hebrew(text, god_list)
    words_list = text.split(SynthDataInfo.word_seperator)
    words_list = [word for word in words_list if len(word.replace(" ", "")) > 1]

    lines_texts, image_texts = split_to_lines(words_list, data_info_list, data_info_probs)

    return lines_texts, image_texts

def split_text_to_files(text, num_lines_in_file, data_info_list, data_info_probs, god_list):

    # read file and convert to valid text of appropriate size

    text = text.replace("\n", " ")
    num_lines_in_file = int(num_lines_in_file)
    texts, image_texts = create_line_texts(text, data_info_list, data_info_probs, god_list)
    texts = ["\n".join(texts[num_lines_in_file * i:num_lines_in_file * (i + 1)]) for i in
             range(len(texts) // num_lines_in_file + 1)]
    image_texts = ["\n".join(image_texts[num_lines_in_file * i:num_lines_in_file * (i + 1)]) for i in
             range(len(image_texts) // num_lines_in_file + 1)]
    return texts, image_texts


def is_segmentation_correct(line_segs, num_lines):
    if len(line_segs) != num_lines:
        return False
    line_hights = np.array([seg.shape[0] for seg in line_segs.values()])
    med = np.median(np.array(line_hights))
    if not all([hight > (med/3) for hight in line_hights]):
        print(line_hights)
    return all([hight > (med/3) for hight in line_hights])



def create_images_per_path(orig_path, base_images_path, base_text_path, num_lines_in_file, font_dir,
                           data_info_list, data_info_probs, god_list, base_path_to_save=None, tmp_workplace='./tmp',
                           do_size_rand=True):
    save_bad = True
    outdata = {}
    try:
        file = pathlib.Path(orig_path)
        text = file.read_text()
        text = re.sub(r'\([^)]*\)', '', text)
        texts, image_texts = split_text_to_files(
            text,
            num_lines_in_file,
            data_info_list, data_info_probs, god_list)

    except Exception as e:
        print('Error while parsing text in file {} to lines for synthesis.'.format(orig_path))
        traceback.print_exc()
        return []
    os.makedirs(tmp_workplace, exist_ok=True)
    file = pathlib.Path(orig_path)
    file_base_name = ('.').join(str(file.name).split('.')[:-1])

    for cur_im_num, (text, im_text) in enumerate(zip(texts, image_texts)):
        # save images
        rel_dir = pathlib.Path(file_base_name) / str(cur_im_num // 1000)
        rel_path = rel_dir / str(cur_im_num)
        path = pathlib.Path(base_images_path) / rel_path
        path.parents[0].mkdir(parents=True, exist_ok=True)
        for fid, font in enumerate(data_info_list[0].font_names):
            try:
                cur_path_no_font = str(path.absolute())
                if len(data_info_list[0].font_names) > 1:
                    cur_path = cur_path_no_font + '_' + str(font)
                if do_size_rand:
                    font_weight = str(np.random.randint(6))
                    font_stretch = str(np.random.randint(9))
                    letter_spacing = "'"+str(np.random.randint(3)) + "'"
                    font_size = "'"+ str(np.random.randint(6)) + "'"
                else:
                    raise NotImplementedError()
                    font_weight = str(3)
                    font_stretch = str(4)
                    letter_spacing = "'" + str(1) + "'"
                    font_size = "'" + str(2) + "'"
                out_im_path = str(cur_path) + '.png'
                run_args = ['../TextRender/bin/main', im_text, out_im_path, font_dir, font, font_weight,
                            font_stretch, letter_spacing, font_size]
                # lock.acquire()
                subprocess.run(run_args, check=True)
                # lock.release()
                # save text
                failed = False
                if data_info_list[0].multi_line:
                    try:
                        line2im, sine_image = im2lines(out_im_path, tmp_workplace=tmp_workplace, verbose=False, max_theta_diff=0.7, rand_sine=True)

                    except Exception as e:
                        print("exception on image: {}".format(out_im_path))
                        traceback.print_exc()
                        failed = True
                    if not is_segmentation_correct(line2im, len(text.split("\n"))) or failed:
                        if save_bad:
                            line_im_path_base = os.path.join(tmp_workplace, time.strftime("%Y%m%d-%H%M%S") + "_" + str(i) + "_" + str(font))
                            for i in range(len(line2im)):
                                line_im_path = line_im_path_base + "_" + str(i) + "_" + str(font)
                                line_im = line2im[i]
                                io.imsave(str(line_im_path) + ".png", line_im)
                        print("Image: {} after sine - Found {} lines, but there are {} lines.".format(out_im_path, len(line2im),
                                                                                           len(text.split("\n"))))
                        try:
                            line2im = im2lines(out_im_path, tmp_workplace=tmp_workplace, verbose=False,
                                                           max_theta_diff=0.7, rand_sine=False)
                        except Exception as e:
                            print("exception on image: {}".format(out_im_path))
                            traceback.print_exc()
                            continue
                        if not is_segmentation_correct(line2im, len(text.split("\n"))):
                            io.imsave(out_im_path, sine_image)
                            for i, (text_line, im_text_line) in enumerate(zip(text.split("\n"), im_text.split("\n"))):
                                line_im_path = cur_path_no_font + "_" + str(i) + "_" + str(font)+".png"
                                run_args = ['../TextRender/bin/main', im_text_line, line_im_path, font_dir, font, font_weight,
                                            font_stretch, letter_spacing, font_size]
                                subprocess.run(run_args, check=True)
                                outdata[str(rel_path) + "_" + str(i)] = text_line
                            print("Image: {} - Found {} lines, but there are {} lines.".format(out_im_path, len(line2im), len(text.split("\n"))))
                        else:
                            os.remove(out_im_path)
                            lines_texts = text.split("\n")
                            for i in range(len(line2im)):
                                line_im_path = cur_path_no_font + "_" + str(i) + "_" + str(font)
                                line_im = line2im[i]
                                io.imsave(str(line_im_path) + ".png", line_im)
                                outdata[str(rel_path) + "_" + str(i)] = lines_texts[i]

                    else:
                        os.remove(out_im_path)
                        lines_texts = text.split("\n")
                        for i in range(len(line2im)):
                            line_im_path = cur_path_no_font + "_" + str(i) + "_" + str(font)
                            line_im = line2im[i]
                            io.imsave(str(line_im_path) +".png", line_im)
                            outdata[str(rel_path) + "_" + str(i)] = lines_texts[i]
                else:
                    if base_path_to_save is not None:
                        outdata[str(rel_path)] = text
                    else:
                        outdata[str(rel_path)] = text
            except Exception as e:
                print("Error while writing to path: {}".format(path))
                print('writing tibetan text:')
                print(text)
                print('original text path is: {}'.format(orig_path))
                traceback.print_exc()
    out_lines = [key + '   *   ' + val + '\n' for key, val in outdata.items()]
    return out_lines


def init_multi_p(in_lock):
    global lock
    lock = in_lock

def create_all_images(text_dir, outdir, data_info_list, data_info_probs, data_info_name, do_size_rand, god_list,
                      tmp_workplace=tmp_workplace):
    num_lines_in_file = 5 if data_info_list[0].multi_line else 1
    font_dir = str(pathlib.Path('extra/Fonts').absolute())
    base_path = pathlib.Path(outdir)
    base_path.mkdir(parents=True, exist_ok=True)
    base_images_path = base_path / 'Images'
    base_images_path.mkdir(parents=False, exist_ok=True)
    base_text_path = base_path / 'Text'
    base_text_path.mkdir(parents=False, exist_ok=True)
    all_texts = glob.glob(text_dir + "/*")
    all_texts = all_texts
    out_file = base_path / 'data.txt'

    out_file = str(out_file)
    if os.path.exists(out_file):
        raise Exception("Error: output file exists already. If you want to override, please delete it first:\n {}".format(
            out_file
        ))

    # save data creation info file
    for i, (data_info, name) in enumerate(zip(data_info_list,data_info_name)):
        with open(str(base_path / 'data_info_{}.pkl'.format(name)), 'wb') as data_inf_f:
            pkl.dump(data_info, data_inf_f)
    with open(str(base_path / 'data_info_probabilities.txt'), 'w') as f:
        f.writelines(["name - {} - prob: {}\n".format(name, prob) for name, prob in zip(data_info_name, data_info_probs)])

    create_images_partial = partial(create_images_per_path, base_images_path=str(base_images_path),
                                  base_text_path=str(base_text_path),
                                  num_lines_in_file=num_lines_in_file,
                                  font_dir=font_dir,
                                data_info_list=data_info_list, data_info_probs=data_info_probs,
                                    do_size_rand=do_size_rand,
                                    god_list=god_list,
                                    tmp_workplace=tmp_workplace)

    l = Lock()
    with Pool(processes=30,initializer=init_multi_p, initargs=(l,)) as p:
        max_ = len(all_texts)
        results = list(tqdm.tqdm(p.imap_unordered(create_images_partial, all_texts), total=max_))
    flatten = lambda l: [item for sublist in l for item in sublist]
    with open(out_file, 'w') as data_f:
        data_f.writelines(flatten(results))

    '''
    out_file_base = out_file_base + '_' + font_name
    path_text_lines = [str(rec[0]) + '   *   ' + str(rec[1]) + '\n' for rec in path2text]
    with open(out_file_base + '.txt', 'w') as f:
        f.writelines(path_text_lines)
    with open(out_file_base + '.pkl', 'wb') as f:
        pickle.dump(path2text, f)
    '''


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_path', dest="input_dir_path",
                        type=str, default=root_dir + 'ocr_datasets/Hebrew/Dataset/Texts/mishna',
                        help=('tibetan_dir'))
    parser.add_argument('--out_dataset_dir', dest="out_dataset_dir", type=str,
                        default=root_dir + 'ocr_datasets/Hebrew/synth/mishna_2',
                        help='Path to directory to save output in. file to save output dataset in (<full image path><seperator><text>\\n) format.')
    parser.add_argument('--tmp_workplace', dest="tmp_workplace", type=str,
                        default=root_dir + 'ocr_datasets/Hebrew/synth/mishna_2/tmp',
                        help='Path to directory to save output in. file to save output dataset in (<full image path><seperator><text>\\n) format.')

    parser.add_argument('--out_name', dest="out_name",
                        type=str,
                        help=('output name'))
    parser.add_argument('--font_names_list', dest="font_names_list", type=str,
                        help='Path to file containing font names.',
                        default='font_names.txt')
    parser.add_argument('--god_font_names_list', dest="god_font_names_list", type=str,
                        help='Path to file containing font names.',
                        default='god_fonts.txt')
    parser.add_argument('--god_names_list', dest="god_names_list", type=str,
                        help='Path to file containing font names.',
                        default='god_names.txt')
    parser.add_argument('--font_dir', dest="font_dir", type=str, default='Fonts',
                        help='Path to directory containing all fonts ttf files.')
    parser.add_argument('--remove_letter_sizes_rand', default=False, action='store_true')
    parser.add_argument('--no_multi_line', default=False, action='store_true')
    args = parser.parse_args()

    with open(args.font_names_list, 'r') as f:
        font_names = f.readlines()
    font_names = [name.replace("\n", "") for name in font_names]

    with open(args.god_names_list, 'r') as f:
        god_names = f.readlines()
    god_names = [name.replace("\n", "") for name in god_names]

    with open(args.god_font_names_list, 'r') as f:
        god_font_names_list = f.readlines()
    god_font_names_list = [name.replace("\n", "") for name in god_font_names_list]


    if args.remove_letter_sizes_rand:
        do_size_rand = False
    else:
        do_size_rand = True

    if args.no_multi_line:
        multi_line = False
    else:
        multi_line = True

    data_info_list = [SynthDataInfo(multi_line=multi_line, font_names=font_names, god_fonts_names=god_font_names_list)]
    data_info_probs = [1]
    data_info_name = ['hebrew_data']



    os.makedirs(args.out_dataset_dir, exist_ok=True)
    create_all_images(args.input_dir_path, args.out_dataset_dir, data_info_list, data_info_probs, data_info_name, do_size_rand,
                      god_names)
    #pikle_to_text(out_dataset_file)
