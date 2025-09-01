"""
Goal:
    A class similar to `seg_dataset`, but for an image volume instead of a folder containing images

"""
import torch
from torch.utils.data import Dataset
import numpy as np
import data.seg_transforms as dt
from data.seg_transforms import Compose as MyCompose
import matplotlib.pyplot as plt

from read_volume_and_compute_enface import get_volume_and_enface_image


class segListVolume(Dataset):
    def __init__(self, volume_path, transforms):
        self.phase = "predict"
        self.volume_path = volume_path
        self.transforms = transforms
        if self.transforms != []:
            if not isinstance(self.transforms, MyCompose):
                self.transforms = MyCompose(self.transforms)
        if volume_path.split('.')[-1] == "img":
            # self.vol = np.load(volume_path)  # Shape (200, 1024, 200) - y * z * x
            self.vol, _ = get_volume_and_enface_image(volume_path)
        else:
            raise Exception("NOT SUPPORTED FILE TYPE")

    def __getitem__(self, index):
        """
        Output:
            (image,imt,imn), in which:
            imt: original image
            image: transformed image (e.g., standardized image) - Its shape must be CxHxW
            imn: image file name
        """

        imt = torch.from_numpy(self.vol[index, :, :])      # Original image as a Pytoch Tensor of size 200x1024
        if imt.ndim < 4:
            imte = np.expand_dims(imt, axis=0)    # 200x1024 => 1x200x1024
        if imte.ndim < 4:
            imte = np.expand_dims(imte, axis=0)    # 1x200x1024 => 1x1x200x1024
        if self.transforms != []:
            # Note: if self.transforms' input has 3 dimensions, its output has 2 dimensions.
            # Therefore, it is better to use a 4-D input, then its output will be 3-D (e.g., 1x200x1024)
            image = self.transforms(*imte)[0]          # The output is a Torch tensor with a size of 1x200x1024
        else:
            image = imte
            if image.ndim == 4:
                image = np.squeeze(image, axis=0)
        imn = f'slice_{index + 1}.png'
        return  (image, imt, imn)

    def __len__(self):
        return self.vol.shape[0]


if __name__ == "__main__":

    PIXEL_MEAN = [0.14162449977018857] * 3
    PIXEL_STD = [0.09798050174952816] * 3
    normalize = dt.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    # t = [dt.ToTensor(), normalize]
    t = []

    vol_path = "e:/data/NYU-OCTseg-dataset/nyu_for_annotation/Normal-ONH-000420-2010-04-22-10-53-54-OD.npy"

    # vol_data = segListVolume(vol_path, dt.Compose(t))
    vol_data = segListVolume(vol_path, t)

    data = vol_data[100]
    plt.imshow(data[1].numpy(), cmap="gray")
    plt.title(f"Number of images {len(vol_data)} in this volume")
    plt.show()

    assert data[0].ndim == 3, "The data[0] must be 3-D"
    # this is what we may use as a input to a CNN.