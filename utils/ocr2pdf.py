# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--image_list_filename', type=str,
                        default="/Users/alexey.p/thesis/tmp/tibetan_ocr_results/test_val_orig_0_im.txt")
    parser.add_argument('-p', '--pred_filename', type=str,
                        default="/Users/alexey.p/thesis/tmp/tibetan_ocr_results/test_val_orig_0_pred_no_stopchars.txt")
    parser.add_argument('-s', '--sorted_pred_filename', type=str,
                        default="/Users/alexey.p/thesis/tmp/tibetan_ocr_results/test_val_orig_0_pred_no_stopchars_sorted.txt")
    parser.add_argument('-b', '--images_base_dir', type=str,
                        default="/Users/alexey.p/thesis/tmp/tibetan_ocr_results/LineImages/")
    parser.add_argument('-o', '--output_image_filename', type=str,
                        default="1k_sorted.pdf")
    parser.add_argument('-f', '--font', type=str,
                        default="Qomolangma-Drutsa.ttf")
    args = parser.parse_args()

    image_list_file = open(args.image_list_filename,'r')
    pred_list_file = open(args.pred_filename,'r')
    image_names = image_list_file.readlines()
    image_indices = [(int(os.path.basename(im_name).split('_')[0]), int(os.path.basename(im_name).split('_')[2].split('.')[0])) for im_name in image_names]
    preds = pred_list_file.readlines()
    image_list_file.close()
    pred_list_file.close()

    sorted_pred_list_file = open(args.sorted_pred_filename,'w')

    imagelist = []
    amount = -1
    sorted_pairs = sorted(zip(preds[:], image_names[:], image_indices[:]), key=lambda x: x[2])[:amount]

    for pred, im_name, im_index in sorted_pairs:
        image_filename = args.images_base_dir + os.path.basename(im_name).rstrip()
        # print(image_filename)
        cover = Image.open(image_filename)
        imagelist.append(cover)
        print(pred)

        pred_img = Image.new('RGB', (cover.size[0] + 100, cover.size[1] + 100), 'white')
        im_draw = ImageDraw.Draw(pred_img)
        font_name = args.font
        font_size = 64
        font = ImageFont.truetype(font_name, font_size)
        im_draw.text((400, 20), pred.decode('utf8'), fill=(0, 0, 0), font=font)
        imagelist.append(pred_img)

        sorted_pred_list_file.write(pred)
        # cover.show()

    sorted_pred_list_file.close()
    imagelist[0].save(args.output_image_filename, "PDF", resolution=100.0, save_all=True, append_images=imagelist[1:])
