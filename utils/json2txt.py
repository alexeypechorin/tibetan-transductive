import json
import os
from argparse import ArgumentParser
from glob import glob

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--input_texts_expression', type=str,
                        help='folder containing json files for text extraction',
                        default='/home/wolf/alexeyp/ocr_datasets/wiener_transcribed/wiener_zahi_clean/*gt.json')
    args = parser.parse_args()
    for filename in glob(args.input_texts_expression):
        # print(filename)
        file = open(filename)
        words_dict = json.load(file)
        last_bbox_bound = 0
        words_string = ""
        for word_data in words_dict:
            bbox_bound = word_data['bbox'][0]
            word = word_data['word']
            if bbox_bound >= last_bbox_bound:
                separator = ' '
            else:
                separator = '\n'
            last_bbox_bound = bbox_bound
            words_string = words_string + separator + word
        file.close()
        pre, ext = os.path.splitext(filename)
        new_filename = pre + ".txt"
        # print(new_filename)
        print(words_string)
        new_file = open(new_filename, 'w+', encoding='utf8')
        new_file.write(words_string)
        new_file.close()
        # print(words_string)
