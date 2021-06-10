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
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize

print(os.listdir(join('../dataset/ribfrac/train', 'ct')))

ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)

ct_array = sitk.GetArrayFromImage(ct)
seg_array = sitk.GetArrayFromImage(seg)

ct_array = ct_array / self.args.norm_factor
ct_array = ct_array.astype(np.float32)