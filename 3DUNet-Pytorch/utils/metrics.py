import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)


class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)

def FROC(gt_label, pred_label, key_fp_list=(0.5, 1, 2, 4, 8)):
    """

    :param gt_label: numpy[C,:,:,:]
    :param pred_label: numpy[C,:,:,:]
    :param key_fp_list:
    :return:
    """
    items = gt_label.shape[0]
    Fp = []
    Recall = []
    for i in range(items):
        gt_label_i = np.argmax(gt_label[i,:], axis=0).astype(np.uint8)
        pred_label_i = pred_label[i,:].astype(np.uint8)

        # GT and prediction must have the same shape
        assert gt_label_i.shape == pred_label_i.shape, \
            "The prediction and ground-truth have different shapes. gt:" \
                f" {gt_label.shape} and pred: {pred_label.shape}."

        # binarize the GT and prediction
        gt_bin = (gt_label_i > 0).astype(np.uint8)
        pred_bin = (pred_label_i > 0).astype(np.uint8)

        intersection = np.logical_and(gt_bin, pred_bin)
        union = np.logical_or(gt_bin, pred_bin)

        EPS = 1e-8
        TP = intersection.sum()
        FP = (union > 0).sum() - TP
        fp = (FP + EPS) / ((pred_bin > 0).sum() + EPS)
        recall = (TP + EPS) / ((gt_bin > 0).sum() + EPS)
        Fp.append(fp)
        Recall.append(recall)

    key_recall = [_interpolate_recall_at_fp(Fp, Recall, key_fp)
                  for key_fp in key_fp_list]
    avg_recall = np.mean(key_recall)

    return avg_recall


def _interpolate_recall_at_fp(fp, recall, key_fp):
    fp = np.array(fp)
    recall = np.array(recall)
    less_fp = fp[fp < key_fp]
    less_recall = recall[fp < key_fp]
    more_fp = fp[fp >= key_fp]
    more_recall = recall[fp >= key_fp]

    # if key_fp < min_fp, recall = 0
    if len(less_fp) == 0:
        return 0

    # if key_fp > max_fp, recall = max_recall
    if len(more_fp) == 0:
        return recall.max()

    fp_0 = np.max(less_fp)
    fp_1 = np.min(more_fp)
    recall_0 = np.max(less_recall)
    recall_1 = np.min(more_recall)
    recall_at_fp = recall_0 + (recall_1 - recall_0) \
                   * ((key_fp - fp_0) / (fp_1 - fp_0 + 1e-8))

    return recall_at_fp
