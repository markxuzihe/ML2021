import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from sklearn import metrics
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects

import matplotlib.pylab as plt
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
    gt_label_i = gt_label
    pred_label_i = pred_label

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


def _remove_low_probs(pred, prob_thresh):
    pred = np.where(pred > prob_thresh, 1, 0)

    return pred


def _remove_small_objects(pred, size_thresh):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred


def post_process(pred, prob_thresh=0.1, size_thresh=10):

    # remove connected regions with low confidence
    pred = _remove_low_probs(pred, prob_thresh)

    # remove small connected regions
    pred = _remove_small_objects(pred, size_thresh)

    return pred


def FROC2(gt_label, pred_label, key_fp_list=(0.5, 1, 2, 4, 8)):
    """

    :param gt_label: numpy[C,:,:,:]
    :param pred_label: numpy[C,:,:,:]
    :param key_fp_list:
    :return:
    """
    gt_num = sum(gt_label)
    total_num = len(pred_label)
    # print(gt_num)
    # print(type(gt_label))
    # print(type(pred_label))
    # print(gt_label.shape)
    # print(pred_label.shape)
    # print(np.sum(pred_label))

    fpr, tpr, thresholds = metrics.roc_curve(gt_label, pred_label, pos_label=1)
    print(len(fpr))
    print(len(tpr))
    print(len(thresholds))

    fps = fpr * (total_num -  gt_num) / total_num
    sens = tpr

    plt.plot(fps, sens, color='b', lw=2)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.125, 8])
    plt.ylim([0, 1.1])
    plt.xlabel('Average number of false positives per scan') #????????????fpr
    plt.ylabel('True Positive Rate')  #????????????tpr
    plt.title('FROC performence')
    plt.show()


    #return avg_recall