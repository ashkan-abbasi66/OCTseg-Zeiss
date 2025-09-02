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
import os

def get_volume_and_enface_image(oct_filepath):
    """
    Reading Cirrus HD-OCT (Zeiss, Dublin, CA, USA) image volumes from `.img` files

    vol: (B-scan index (y), depth axis (z), lateral axes (x)) - 200 x 1024 x 200
    enface: A 200 x 200 image obtained by averaging along z axis.
    """
    if not os.path.exists(oct_filepath):
        return -1

    with open(oct_filepath, 'rb') as f:
        numpy_data = np.fromfile(f, np.dtype('B'))

    vol = np.reshape(numpy_data, (200, 1024, 200))
    vol = np.flip(vol, axis=1)  # For Enface image, if you make this line a comment, it won't change the output.
                                  # In fact, flipping along Z axis does not change anything except that it changes
                                  # the order of B-scans.
    vol = np.flip(vol, axis=2)

    enface = np.mean(vol, axis=1)

    return vol, enface


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

    # Check this for two volume




    PIXEL_MEAN = [0.14162449977018857] * 3
    PIXEL_STD = [0.09798050174952816] * 3
    normalize = dt.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    # t = [dt.ToTensor(), normalize]
    t = []

    vol_path = "../my-example-datasets/example-Optic Disc Cube 200x200-OS-cube_z.img"

    # vol_data = segListVolume(vol_path, dt.Compose(t))
    vol_data = segListVolume(vol_path, t)

    data = vol_data.vol
    plt.imshow(data[10],cmap = "gray")
    plt.title(f"One slice from a volume with {len(vol_data)} images")

    assert data.ndim == 3, "The data[0] must be 3-D"
    # this is what we may use as a input to a CNN.


    # TEST reading OCT
    vol, enface_image = get_volume_and_enface_image(vol_path)
    plt.figure()
    plt.imshow(enface_image, cmap="gray")
    plt.title(f"Enface image's shape:{enface_image.shape}")
    plt.show()