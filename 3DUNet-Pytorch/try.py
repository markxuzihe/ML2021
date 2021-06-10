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

ct = sitk.ReadImage('../dataset/fixed_train/ct/RibFrac1-image.nii.gz', sitk.sitkInt16)
seg = sitk.ReadImage('../dataset/fixed_train/label/RibFrac1-label.nii.gz', sitk.sitkUInt8)

ct_array = sitk.GetArrayFromImage(ct)
seg_array = sitk.GetArrayFromImage(seg)

flag=0
for i in range(len(seg_array)):
    for j in range(len(seg_array[i])):
        for k in seg_array[i][j]:
            if k!=0:
                print(k)
