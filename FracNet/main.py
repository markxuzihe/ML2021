from functools import partial
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from dataset.fracnet_dataset import FracNetTrainDataset
from dataset import transforms as tsfm
from utils.metrics import dice, recall, precision, fbeta_score
from model.unet import UNet
from model.losses import MixLoss, DiceLoss


def main(args):
    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir

    batch_size = 4
    num_workers = 4
    criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)

    thresh = 0.1
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)
    device = torch.device('cuda:1')
    model = UNet(1, 1, first_out_channels=16).to(device)
    # model = nn.DataParallel(model.to(device))
    try:
        model_weights = torch.load("result/latest_model_weights.pth")
        model.load_state_dict(model_weights)
        print("weights loading")
    except:
        print("load failed")
        pass
    model = nn.DataParallel(model, device_ids=[1])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #loss = criterion
    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
        transforms=transforms)
    dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)
    # ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
    #     transforms=transforms)
    # dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
    #     num_workers)

    for epoch in range(args.epochs):
        if epoch==20:
            args.lr = 0.0001
        print("epoch: {}".format(epoch+1))
        train_dice = 0
        train_loss = 0
        for idx, (data, target) in tqdm(enumerate(dl_train), total=len(dl_train)):
            data, target = data.float(), target.float()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            train_dice += dice(output,target)
            train_loss += float(loss)
        train_dice /= len(dl_train)
        train_loss /= len(dl_train)
        torch.save(model.module.state_dict(), "result/latest_model_weights.pth")
        print("train dice: {}, train loss: {}".format(train_dice, train_loss))




    if args.save_model:
        torch.save(model.module.state_dict(), "result/model_weights.pth")


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", default="../dataset/train/ct",
        help="The training image nii directory.")
    parser.add_argument("--train_label_dir", default="../dataset/train/label",
        help="The training label nii directory.")
    parser.add_argument("--val_image_dir", default="../dataset/val/ct",
        help="The validation image nii directory.")
    parser.add_argument("--val_label_dir", default="../dataset/val/label",
        help="The validation label nii directory.")
    parser.add_argument("--save_model", default=True,
        help="Whether to save the trained model.")
    parser.add_argument("--epochs", default=50,
        help="...")
    parser.add_argument("--lr", default=0.0001,
        help="...")

    args = parser.parse_args()

    main(args)
