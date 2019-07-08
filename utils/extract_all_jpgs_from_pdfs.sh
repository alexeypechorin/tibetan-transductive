#!/usr/bin/env bash



find /specific/disk1/home/alexeyp/Tibetan_Transductive/Data/Tibetan_test_images/ -name "*.pdf" -exec python /specific/disk1/home/alexeyp/Tibetan_Transductive/Src/utils/extract_jpg_from_pdf.py --input_pdf_filename \{\} --output_folder \{\}_images/Images \;
