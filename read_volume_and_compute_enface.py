import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

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


if __name__ == '__main__':
    oct_filepath = "example-Optic Disc Cube 200x200-OS-cube_z.img"

    vol, enface_image = get_volume_and_enface_image(oct_filepath)
    plt.figure()
    plt.imshow(enface_image, cmap="gray")
    plt.title(f"Enface image's shape:{enface_image.shape}")

    plt.show()