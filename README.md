# EE228-Final-Project: Medical Image Cutting On RibFrac Dataset

Our main code structure is referred to [3DUNet-Pytorch](https://github.com/lee-zq/3DUNet-Pytorch) and [FracNet](https://github.com/M3DV/FracNet).

## How to run our codes in two different frameworks

### In [3DUNet-Pytorch](./3DUNet-Pytorch) framework:

1. Create environment and install packages

```
pytorch >= 1.1.0
torchvision
SimpleITK
Tensorboard
Scipy
```

2. Preprocess the dataset (you may need to change the dataset path first, and change configs in [config.py](./3DUNet-Pytorch/config.py))

```
python preprocess_LiTS.py
```

3. Begin to train (also change configs in [config.py](./3DUNet-Pytorch/config.py))

```
python train.py
```

### In [FracNet](./FracNet) framework:

1. Create environment and install packages

```
SimpleITK==1.2.4
fastai==1.0.59
fastprogress==0.1.21
matplotlib==3.1.3
nibabel==3.0.0
numpy>=1.18.5
pandas>=0.25.3
scikit-image==0.16.2
torch==1.4.0
tqdm==4.38.0
```

2. Begin to train

You can choose to run [main.py](./FracNet/main.py) or use [train_final.ipynb](./FracNet/train-final.ipynb) to get a more detailed training process.

## Experiment Environment

- Ubuntu 16.04
- NVIDIA GTX 1080TI

## How to reproduce our work

The main difference between the two frameworks is the method of processing data.

In [3DUNet-Pytorch](./3DUNet-Pytorch), we just simply cut off the origin image in the z-axis direction, while in [FracNet](./FracNet) framework we truncate in totally 3 directions. Moreover, in [FracNet](./FracNet) we also generate some negative samples for model to learn, which also gives us a better result.

Due to the limit of time and computing resources, we haven't get a fully trained model yet. Our best model is uploaded as [best_model.pth](./FracNet/result/best_model.pth), to reproduce our work, you only need to run [predict.py](./FracNet/predict.py) to get the prediction results. 