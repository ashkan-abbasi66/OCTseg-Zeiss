import numpy as np
import torch
import os
import os.path as osp
import cv2
import scipy.misc as misc
import shutil
from skimage import measure
import math
import traceback
from sklearn import metrics
import zipfile


def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        self.avg = self.sum / self.count


def save_model(state, is_best, model_path):
    model_latest_path = osp.join(model_path,'model_latest.pth.tar')   
    torch.save(state, model_latest_path)
    if is_best:
        model_best_path = osp.join(model_path,'model_best.pth.tar')
        shutil.copyfile(model_latest_path, model_best_path)


def save_dice_single(is_best, filename='dice_single.txt'):
    if is_best:
        shutil.copyfile(filename, 'dice_best.txt')


def compute_dice(ground_truth, prediction, n_classes=11):
    """
    Dice Score = 2 * (TP / (TP + FP + FN))
    A Dice score of 1 indicates perfect overlap
    A score of 0 implies no overlap
    """
    ground_truth = ground_truth.flatten()   # (1, 1024, 200) => (204800,)
    prediction = prediction.flatten()
    try:
        # ret = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ret = [0.5]*n_classes
        for i in range(n_classes):
            mask1 = (ground_truth == i)  # total number of pixels in the ground truth that are relevant to class i
            mask2 = (prediction == i)    # total number of predictions for class i
            if mask1.sum() != 0:
                TP = ((mask1 * (ground_truth == prediction)).sum())
                ret[i] = float(2 * TP / (mask1.sum() + mask2.sum()))
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret


def compute_pa(ground_truth, prediction, n_classes=11):
    """
    Pixel Accuracy = (TP / Total pixels) * 100 %
    Returns: (TP / Total pixels)
    """
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        # ret = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ret = [0.5] * n_classes
        for i in range(n_classes):
            mask1 = (ground_truth == i)
            if mask1.sum() != 0:
                TP = ((mask1 * (ground_truth == prediction)).sum())
                total_pixels = mask1.sum()
                ret[i] = float(TP / total_pixels)
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret


def compute_average_score_for_one_image(ret_d):
    """
    "ret_d" is a list of scores.
    Returns the averages of scores without considering the background class

    Notes:
    - Use this function instead of "compute_single_avg_score"
    - The background's numeric label is assumed to be 0.
    """
    # dice_score = compute_single_avg_score(ret_d)
    dice_score = np.nanmean(np.array(ret_d)[1:])  # all classes except background
    return dice_score


def compute_single_avg_score(ret_seg):  # It does not consider the background class
    NFL_seg, GCL_seg, IPL_seg, INL_seg, OPL_seg, ONL_seg, IS_OS_seg, RPE_seg, Choroid_seg, Disc_seg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if not math.isnan(ret_seg[1]):
        NFL_seg = ret_seg[1]
    if not math.isnan(ret_seg[2]):
        GCL_seg = ret_seg[2]
    if not math.isnan(ret_seg[3]):
        IPL_seg = ret_seg[3]
    if not math.isnan(ret_seg[4]):
        INL_seg = ret_seg[4]
    if not math.isnan(ret_seg[5]):
        OPL_seg = ret_seg[5]
    if not math.isnan(ret_seg[6]):
        ONL_seg = ret_seg[6]
    if not math.isnan(ret_seg[7]):
        IS_OS_seg = ret_seg[7]
    if not math.isnan(ret_seg[8]):
        RPE_seg = ret_seg[8]
    if not math.isnan(ret_seg[9]):
        Choroid_seg = ret_seg[9]
    if not math.isnan(ret_seg[10]):
        Disc_seg = ret_seg[10]
    avg_seg = (NFL_seg + GCL_seg + IPL_seg + INL_seg + OPL_seg + ONL_seg + IS_OS_seg + RPE_seg + Choroid_seg + Disc_seg) / 10
    return avg_seg


# def compute_avg_score(ret_seg):
#     BG, NFL_seg, GCL_seg, IPL_seg, INL_seg, OPL_seg, ONL_seg, IS_OS_seg, RPE_seg, Choroid_seg, Disc_seg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#     n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000001
#     num = np.array(ret_seg).shape[0]
#     for i in range(num):
#         if not math.isnan(ret_seg[i][0]):
#             BG += ret_seg[i][0]
#             n0 += 1
#         if not math.isnan(ret_seg[i][1]):
#             NFL_seg += ret_seg[i][1]
#             n1 += 1
#         if not math.isnan(ret_seg[i][2]):
#             GCL_seg += ret_seg[i][2]
#             n2 += 1
#         if not math.isnan(ret_seg[i][3]):
#             IPL_seg += ret_seg[i][3]
#             n3 += 1
#         if not math.isnan(ret_seg[i][4]):
#             INL_seg += ret_seg[i][4]
#             n4 += 1
#         if not math.isnan(ret_seg[i][5]):
#             OPL_seg += ret_seg[i][5]
#             n5 += 1
#         if not math.isnan(ret_seg[i][6]):
#             ONL_seg += ret_seg[i][6]
#             n6 += 1
#         if not math.isnan(ret_seg[i][7]):
#             IS_OS_seg += ret_seg[i][7]
#             n7 += 1
#         if not math.isnan(ret_seg[i][8]):
#             RPE_seg += ret_seg[i][8]
#             n8 += 1
#         if not math.isnan(ret_seg[i][9]):
#             Choroid_seg += ret_seg[i][9]
#             n9 += 1
#         if not math.isnan(ret_seg[i][10]):
#             Disc_seg += ret_seg[i][10]
#             n10 += 1
#     BG /= n0
#     NFL_seg /= n1
#     GCL_seg /= n2
#     IPL_seg /= n3
#     INL_seg /= n4
#     OPL_seg /= n5
#     ONL_seg /= n6
#     IS_OS_seg /= n7
#     RPE_seg /= n8
#     Choroid_seg /= n9
#     Disc_seg /= n10
#     avg_seg = (NFL_seg + GCL_seg + IPL_seg + INL_seg + OPL_seg + ONL_seg + IS_OS_seg + RPE_seg + Choroid_seg + Disc_seg) / 10
#     return avg_seg, NFL_seg, GCL_seg, IPL_seg, INL_seg, OPL_seg, ONL_seg, IS_OS_seg, RPE_seg, Choroid_seg, Disc_seg