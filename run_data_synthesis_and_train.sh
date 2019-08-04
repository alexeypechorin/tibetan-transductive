#!/usr/bin/env bash
cd data_preperation;

python 1_prepare_orig_images.py;
python 2_prepare_synth_images.py;
python 3_create_class_dict.py;

cd ..;

CUDA_VISIBLE_DEVICES=0 train.py --do-test-vat True --vat-epsilon 0.5 --vat-xi 1e-6 --vat-sign True --vat-ratio 10. \
--output-dir '../Output/transductive_vat' --do-lr-step True
