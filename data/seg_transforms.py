import numbers
import random
import numpy as np
from PIL import Image, ImageOps
import torch

class Label_Transform(object):
    """
    Converts pixel values (labels) in the mask image to output labels used
    in the segmentation model.

    Pixel values in the mask image and their corresponding output labels:
    Background      0          0
    RNFL            26         1
    GCL             51         2
    IPL             77         3
    INL             102        4
    OPL             128        5
    ONL             153        6
    IS/OS           179        7
    RPE             204        8
    Choroid         230        9
    Optic disc      255        10

    Total number of classes = Background + 10 = 11
    """
    def __init__(self,label_pixel=(26, 51, 77, 102, 128, 153, 179, 204, 230, 255)):
        self.label_pixel = label_pixel

    def __call__(self, image, label, *args):
        label = np.array(label)
        for i in range(len(self.label_pixel)):
            label[label == self.label_pixel[i]] = i+1

        return image,Image.fromarray(label)


class Label_Transform_NYUPITT(object):
    """
    Converts pixel values (labels) in the mask image to output labels used
    in the segmentation model.

    Pixel values in the mask image and their corresponding output labels:
    RNFL            0          1
    GCL+IPL         30         2
    INL             60         3
    OPL             90         4
    ONL             120        5
    IS              150        6
    OS              180        7
    RPE             210        8
    Background      240        0

    Total number of classes = Background + 8 = 9
    """
    def __init__(self,label_pixel=(0, 30, 60, 90, 120, 150, 180, 210), is_mac=False):
        self.is_mac = is_mac
        if is_mac:
            self.label_pixel = label_pixel[:-1]
        else:
            self.label_pixel = label_pixel

    def __call__(self, image, label, *args):
        label = np.array(label)
        for i in range(len(self.label_pixel)):
            label[label == self.label_pixel[i]] = i + 1
        if self.is_mac:
            label[label == 210] = 0  # Background class
        else:
            label[label == 240] = 0  # Background class

        return image,Image.fromarray(label)

import torch

# class Label_Transform_NYUPITT_v2(object):
#     """
#     See Label_Transform_NYUPITT
#     """
#     def __init__(self,label_pixel=(0, 30, 60, 90, 120, 150, 180, 210), is_mac=False):
#         self.is_mac = is_mac
#         if is_mac:
#             self.label_pixel = label_pixel[:-1]
#         else:
#             self.label_pixel = label_pixel
#
#     def __call__(self, image, label, *args):
#         label = np.array(label, dtype=np.uint8)
#         for i in range(len(self.label_pixel)):
#             label[label == self.label_pixel[i]] = i + 1
#         if self.is_mac:
#             label[label == 210] = 0
#         else:
#             label[label == 240] = 0
#         # return label as torch tensor directly
#         return image, torch.from_numpy(label).long()


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, image, label=None):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        if label is None:
            return image,
        else:
            return image, label


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, label=None):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255)
        if label is None:
            return img,
        else:
            return img, torch.LongTensor(np.array(label, dtype=int))


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
