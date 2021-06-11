
from dataset.dataset_rib_val import Val_Dataset


from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config

from models import UNet, ResUNet, KiUNet_min, SegNet

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict


def predict(model,test_loader,n_labels):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            tmp_target = target
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 计算froc
            if not args.cpu:
                res = output.cpu()
            else:
                res = output
            print(np.sum(res.numpy()[0,1,:]))
            print(res.numpy()[0,1,:].shape)
            pred = res.numpy()[0,1,:]
            tmp_target = tmp_target.numpy()[0,:]
            tmp_target = tmp_target.flatten()
            pred = pred.flatten()
            print(np.sum(tmp_target))
            print(np.sum(pred))
            metrics.FROC2(tmp_target,pred)
            break


if __name__ == '__main__':
    args = config.args
    args.test_data_path = "../dataset/fixed_val"
    save_path = os.path.join('./experiments', args.save)
    device = torch.device('cpu' if args.cpu else 'cuda:1')
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)

    # model info
    model = UNet(in_channel=1, out_channel=args.n_labels,training=False).to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])
    predict(model,val_loader,2)
    # test_log = logger.Test_Logger(save_path,"test_log")
    # # data info
    # result_save_path = '{}/result'.format(save_path)
    # if not os.path.exists(result_save_path):
    #     os.mkdir(result_save_path)

    # for img_dataset,file_idx in datasets:
    #     test_dice,pred_img = predict_one_img(model, img_dataset, args)
    #     test_log.update(file_idx, test_dice)
    #     sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-'+file_idx+'.gz'))
