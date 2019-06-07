# Transductive Learning for Reading Handwritten Tibetan Manuscripts by Sivan Keret

This software implements transductive learning for unsupervised handwritten character recognition. 
It includes:
- A projective based unsupervised line segmentation algorithm
- Synthetic text generation and data augmentation for HCR training
- CRNN implementation for HCR
- Implementation of three methods for transductive learning for HCR: CycleGan, DANN and VAT 

The repository also includes a new test set containing 167 transcribed images of \emph{bKa’ gdams gsung ’bum} collection. \
The software was tested on this collection and shows promising results.


## Prerequisites

The software has only been tested on Ubuntu 16.04 (x64). CUDA-enabled GPUs are required. 
Tested with Cuda 8.0 and Cudnn 7.0.5

## Installation

### Install text rendering software
1. Install Tibetan fonts:
```console
mkdir ~/.fonts
cp extra/Fonts/* ~/.fonts
sudo fc-cache -fv
``` 
2. Change language settings to allow ASCII text reading:
```console
sudo update-locale LANG=en_US.UTF-8
```
3. Install pre-requisits
```console
sudo apt-get install cairo-dock
sudo apt-get install pango1.0-tests
sudo apt-get install gtk2.0
sudo add-apt-repository ppa:glasen/freetype2
sudo apt update && sudo apt install freetype2-demos
``` 
4. Compile cpp text rendering program:
```console
cd extra/TextRender/c_code
make
cd ../bin
chmod u+x main
```
5. Check that both installation and compilation worked correctly:
- to test font installation run:
```console
bin/main_show_fonts | grep Shangshung
bin/main_show_fonts | grep Qomolangma
```
You should see these four font families:
   - Shangshung Sgoba-KhraChung
   - Shangshung Sgoba-KhraChen
   - Qomolangma-Drutsa
   - Qomolangma-Betsu
- To test compilation run (make sure there's a "test" directory for the result to appear):
```console
bash test.sh
```

### Create Python environment and install dependencies
```console
cd <base project dir>
conda create -n tibetan_hcr python=3.6
source activate tibetan_hcr
pip install -r requirements.txt
cd Src/utils/warp-ctc
mkdir build; cd build
cmake ..
make
cd ../pytorch_binding
python setup.py install
cd ../..
export CFLAGS='-Wall -Wextra -std=c99 -I/usr/local/cuda-8.0/include'
#export CFLAGS='-Wall -Wextra -std=c99 -I/usr/local/stow/cuda-8.0/lib/cuda-8.0/include/'
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .

```

The following image should be created in *Src/data/extra/TextRender/test*:
![rendered test image](./docs/rendering_test.png "Image created by running 'bash test.sh'")

## Data Preparation 
There are three parts to data preparation:
1. Using unsupervised line segmentation to separate test images to lines
2. Rendering synthetic multi line images and separating lines using line segmentation
3. Creating a character lexicon from both training and testing datasets.

We provide two ways to get data from training and testing:
1. Downloading prepared data
2. Instructions and code to prepare data
### Downloading Prepared Validation and Synthesized Train Data
1. Get prepared synthetic data:
    - Download Prepared Data ([google drive link](https://drive.google.com/file/d/1Z_ar_ogYmCN_VFKGsav5nP12AN1_Dn3P/view?usp=sharing))
to *Data/Synthetic*
    - untar file:
    ```console
    tar -xzvf synth_prepared.tar.gz
    ```
2. Get prepared test data-set:
    - Download Prepared Data ([google drive link](https://drive.google.com/file/d/1ulgZVktJmnFMMOIn9zCb8Z_BLJ7J7--T/view?usp=sharing))
    to *Data/Test*
    - untar file:
    ```console
    tar -xzvf test_prepared.tar.gz
    ```
3. Get prepared character lexicon:
- Download character lexicon file ([google drive link](https://drive.google.com/file/d/1TFdYqYYEdpREfcQGfgECMm37czU8wqBw/view?usp=sharing))
    to *Data*
### Preparing Test and Train data
1. Segment test images to lines & create dataset file:
    - Download test images ([google drive link](https://drive.google.com/file/d/1ndn7uquU25J97_DK-V9FuMNN2AL65s1y/view?usp=sharing))
    to *Data/Test*
    - Untar file:
    ```console
    tar -xzvf test_original.tar.gz
    ```
    - Segment images to line and create a dataset file containing line image to text tuples:
    ```console
    cd Src/data_preperation
    python 1_prepare_orig_images.py
    ```
2. Create synthetic images and dataset file:
    - Download texts to synthesize train images ([google drive link](https://drive.google.com/file/d/1igUXljiJcDK7OUM7IQB5iHXC0Az63-o3/view?usp=sharing))
    to *Data/Synthetic*
    - Create synthetic images and a dataset file containing line image to text tuples:
    ```console
    cd Src/data_preperation
    python 2_prepare_synth_images.py
    ```
3. Create character lexicon for both synthetic and original data:
```console
cd Src/data_preperation
python 3_create_class_dict.py
```    
## Training & Testing
Now that you have the dataset prepared and all the prerequisits installed, you can run CRNN training and testing.
To do so, go to _base_project_dir/Src_.
### Training Options
- Transductive VAT
```console
python train.py --do-test-vat True --vat-epsilon 0.5 --vat-xi 1e-6 --vat-sign True --vat-ratio 10. \
--output-dir '../Output/transductive_vat' --do-lr-step True
```
- Transductive Adversarial Domain Adaptation
```console
python train.py --do-test-vat True --vat-epsilon 0.5 --vat-xi 1e-6 --vat-sign True --vat-ratio 10. \
--output-dir '../Output/transductive_vat' --do-lr-step True
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
Please cite the following paper if you are using the code/model in your research paper:
"Transductive Learning for Reading Handwritten Tibetan Manuscripts"

