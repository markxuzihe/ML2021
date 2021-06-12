from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
# for a,b,files in os.walk('../dataset/ribfrac/train/ct'):
#     print(files)

path = "../dataset/fixed_val/mylabel/"

for file in os.listdir(path):
    new_name = file.replace("_pred.nii",".nii")
    os.rename(os.path.join(path,file),os.path.join(path,new_name))


