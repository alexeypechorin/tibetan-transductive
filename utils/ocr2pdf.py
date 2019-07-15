from PIL import Image, ImageDraw
import os

if __name__ == '__main__':
    image_list_filename="/Users/alexey.p/thesis/tmp/tibetan_ocr_results/test_val_orig_0_im.txt"
    pred_filename="/Users/alexey.p/thesis/tmp/tibetan_ocr_results/test_val_orig_0_pred_no_stopchars.txt"
    sorted_pred_filename="/Users/alexey.p/thesis/tmp/tibetan_ocr_results/test_val_orig_0_pred_no_stopchars_sorted.txt"
    images_base_dir = '/Users/alexey.p/thesis/tmp/tibetan_ocr_results/LineImages/'
    output_image_filename = "1k_sorted.pdf"

    image_list_file = open(image_list_filename,'r')
    pred_list_file = open(pred_filename,'r')
    image_names = image_list_file.readlines()
    image_indices = [(int(os.path.basename(im_name).split('_')[0]), int(os.path.basename(im_name).split('_')[2].split('.')[0])) for im_name in image_names]
    preds = pred_list_file.readlines()
    image_list_file.close()
    pred_list_file.close()

    sorted_pred_list_file = open(sorted_pred_filename,'w')

    imagelist = []
    amount = -1
    sorted_pairs = sorted(zip(preds[:], image_names[:], image_indices[:]), key=lambda x: x[2])[:amount]

    for pred, im_name, im_index in sorted_pairs:
        image_filename = images_base_dir + os.path.basename(im_name).rstrip()
        # print(image_filename)
        cover = Image.open(image_filename)
        imagelist.append(cover)
        print(pred)
        sorted_pred_list_file.write(pred)
        # cover.show()

    sorted_pred_list_file.close()
    imagelist[0].save(output_image_filename, "PDF", resolution=100.0, save_all=True, append_images=imagelist[1:])
