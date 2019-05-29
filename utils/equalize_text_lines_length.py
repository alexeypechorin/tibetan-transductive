import json
import os
from argparse import ArgumentParser
from glob import glob

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--input_texts_expression', type=str,
                        help='folder containing txt files for line length equalization',
                        default='/home/alexeyp/ext/Tibetan_Transductive/Data/wiener/Synthetic/TextFull/wiener_text_for_synth1.txt')
    args = parser.parse_args()
    for filename in glob(args.input_texts_expression):
        file = open(filename)
        file_lines = file.readlines()
        lines = [line.strip('\n') for line in file_lines if len(line.strip('\n')) > 0]
        import numpy as np
        lens = np.array([len(l) for l in lines])

        def plot_lines_length_dist(lines):
            lines_stripped = [line.strip('\n') for line in lines if len(line.strip('\n')) > 0]
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(lines_stripped)
            plt.show()


        # plot_lines_length_dist(file_lines)
        max_len = max(lens)
        mean_len = int(lens.mean())
        all_text = ' '.join(lines)
        even_length_text = [all_text[ind:ind+mean_len] for ind in range(0, len(all_text), mean_len)]
        prev = ''
        lines_equalized = []
        for line in even_length_text:
            words = line.split(' ')
            prev += words[0]
            curr = ' '.join(words[1:])
            lines_equalized.append(prev + '\n')
            prev = curr
        lines_equalized.pop(0)
        # plot_lines_length_dist(lines_equalized)

        file.close()
        pre, ext = os.path.splitext(filename)
        new_filename = pre + "_equalized.txt"
        # print(new_filename)
        new_file = open(new_filename, 'w+', encoding='utf8')
        new_file.writelines(lines_equalized)
        new_file.close()
