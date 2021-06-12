
from dataset.dataset_rib_val import Val_Dataset


from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
import SimpleITK as sitk
from models import UNet, ResUNet, KiUNet_min, SegNet

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict


def predict(model,file_path,args):
    model.eval()
    with torch.no_grad():
        file_name_list = []
        files=[]
        for a,b,c in os.walk(file_path+'/ct'):
            files=c
        for file_name in files:
            file_name_list.append(file_name)
        
        for index in range(len(file_name_list)):
            ct = sitk.ReadImage(file_path+'/ct/'+file_name_list[index], sitk.sitkInt16)
            ct = sitk.GetArrayFromImage(ct)
            ct = ct/args.norm_factor
            ct = ct.astype(np.float32)
            ct = torch.FloatTensor(ct).unsqueeze(0)
            
            ct = ct.float()
            ct = ct.to(device)

            output = model(ct)
            print(output.shape)
            pred = output.numpy()[0,1,:]
            my_label = sitk.GetImageFromArray(pred)
            my_label.SetDirection(ct.GetDirection())
            my_label.SetOrigin(ct.GetOrigin())
            my_label.SetSpacing((ct.GetSpacing()[0] * int(1 / 1.0),
                           ct.GetSpacing()[1] * int(1 / 1.0), 1.0))

            print(my_label.shape)
            break

            sitk.WriteImage(my_label,file_path+'/mylabel/'+ file_name_list[index])


if __name__ == '__main__':
    args = config.args
    test_data_path = "../dataset/fixed_val"
    save_path = os.path.join('./experiments', args.save)
    device = torch.device('cuda:1')
    # model info
    model = UNet(in_channel=1, out_channel=args.n_labels,training=False).to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])
    predict(model,test_data_path,args)
