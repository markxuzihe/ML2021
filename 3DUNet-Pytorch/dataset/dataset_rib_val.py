from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import Center_Crop, Compose


class Val_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.filename_list = self.load_file_name_list(args.val_data_path)
        self.filename_list = self.filename_list[:8]

        self.transforms = Compose([Center_Crop(base=16, max_size=args.val_crop_max_size)])

    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        files=[]
        for a,b,c in os.walk(file_path+'/ct'):
            files=c
        for file_name in files:
            file_name_list.append([file_path+'/ct/'+file_name, file_path+'/label/'+file_name.replace('image', 'label')])
        return file_name_list

