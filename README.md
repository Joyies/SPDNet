# SPDNet
## Structure-Preserving Deraining with Residue Channel Prior Guidance (ICCV2021) [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2108.09079) [![GitHub Stars](https://img.shields.io/github/stars/Joyies/SPDNet?style=social)](https://github.com/Joyies/SPDNet)

## Requirements

* Linux Platform
* NVIDIA GPU + CUDA CuDNN
* PyTorch == 0.4.1
* torchvision0.2.0
* [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets)
* Python3.6.0
* imageio2.5.0
* numpy1.14.0
* opencv-python
* scikit-image0.13.0
* tqdm4.32.2
* scipy1.2.1
* matplotlib3.1.1
* ipython7.6.1
* h5py2.10.0

## Training
1. Modify data path in src/data/rainheavy.py and src/data/rainheavytest.py <br/>
datapath/data/\*\*\*.png <br/>
datapath/label/\*\*\*.png
3. Begining training:
```
$ cd ./src/
$ python main.py --save spdnet --model spdnet --scale 2 --epochs 300 --batch_size 16 --patch_size 128 --data_train RainHeavy --n_threads 0 --data_test RainHeavyTest --data_range 1-1800/1-200 --loss 1*MSE  --save_results --lr 5e-4 --n_feats 32 --n_resblocks 3
```

## Test
The pre-trained model can be available at google drive: https://drive.google.com/drive/folders/1ylON5AkJVayoypOXDaUEkYd76LtMF-lB?usp=sharing.

```
$ cd ./src/
$ python main.py --data_test RainHeavyTest  --ext img --scale 2  --data_range 1-1800/1-200 --pre_train ../experiment/spdnet/model/model_best.pt --model spdnet --test_only --save_results --save RCDNet_test
```
All PSNR and SSIM results are computed by using this [Matlab code](https://github.com/csdwren/PReNet/tree/master/statistic), based on Y channel of YCbCr space.

## Datasets
**[Rain200L](https://pan.baidu.com/s/1SR7yULy0VZ_JZ4Vawqs7gg#list/path=%2F)**: 1800 training pairs and 200 testing pairs <br/>
**[Rain200H](https://pan.baidu.com/s/1SR7yULy0VZ_JZ4Vawqs7gg#list/path=%2F)**: 1800 training pairs and 200 testing pairs <br/>
**[Rain800](https://drive.google.com/open?id=0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s)**: 700 training pairs and 100 testing pairs <br/>
**[Rain1200](https://drive.google.com/file/d/1cMXWICiblTsRl1zjN8FizF5hXOpVOJz4/view?usp=sharing)**: 12000 traing paris and 1200 testing pairs <br/>
**[SPA-Data](https://stevewongv.github.io/derain-project.html)**: 638492 training pairs and 1000 testing pairs

## Acknowledgement 
Code borrows from [RCDNet](https://github.com/hongwang01/RCDNet). Thanks for sharing !
