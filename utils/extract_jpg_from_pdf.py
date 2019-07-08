# Extract jpg's from pdf's. Quick and dirty.
import errno
import os
import sys
from argparse import ArgumentParser

# From https://nedbatchelder.com/blog/200712/extracting_jpgs_from_pdfs.html

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--input_pdf_filename', type=str,
                        help='Input PDF filename',
                        default="/specific/disk1/home/alexeyp/Tibetan_Transductive/Data/Tibetan_test_images/sNgags log sun 'byin Rin chen bzang po B.pdf")
    parser.add_argument('-o', '--output_folder', type=str,
                        help='Output folder',
                        default="/specific/disk1/home/alexeyp/Tibetan_Transductive/Data/Tibetan_test_images/sNgags log sun 'byin Rin chen bzang po B_images/Images")
    args = parser.parse_args()

    pdf = open(args.input_pdf_filename, "rb").read()

    startmark = b"\xff\xd8"
    startfix = 0
    endmark = b"\xff\xd9"
    endfix = 2
    i = 0

    njpg = 0
    while True:
        istream = pdf.find(b"stream", i)
        if istream < 0:
            break
        istart = pdf.find(startmark, istream, istream + 20)
        if istart < 0:
            i = istream + 20
            continue
        iend = pdf.find(b"endstream", istart)
        if iend < 0:
            raise Exception("Didn't find end of stream!")
        iend = pdf.find(endmark, iend - 20)
        if iend < 0:
            raise Exception("Didn't find end of JPG!")

        istart += startfix
        iend += endfix
        print("JPG %d from %d to %d" % (njpg, istart, iend))
        jpg = pdf[istart:iend]
        image_filename = "-%d.jpg" % njpg
        full_image_filename = os.path.join(args.output_folder, image_filename)
        if not os.path.exists(args.output_folder):
            try:
                os.makedirs(args.output_folder)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        jpgfile = open(full_image_filename, "wb")
        jpgfile.write(jpg)
        jpgfile.close()

        njpg += 1
        i = iend