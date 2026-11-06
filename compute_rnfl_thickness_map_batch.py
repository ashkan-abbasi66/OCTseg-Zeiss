"""
Module for computing RNFL thickness maps from OCT volumes using a pretrained segmentation model.

This script processes OCT volume files (.img) to:
1. Perform retinal layer segmentation using a pretrained U-Net model
2. Generate RNFL thickness maps from the segmentation results
3. Save the segmentation volume (.npy), the RNFL thickness map in micrometers (.npy),
   and a color-coded thickness map with a colorbar (.png)

RNFL thickness is computed in physical units: the number of RNFL pixels in each A-scan
multiplied by the axial pixel size. As in OCT-ONH, the pixel spacing is assumed from the
Cirrus HD-OCT "Optic Disc Cube 200x200" geometry (6 x 6 x 2 mm) and is not read from the file.

The script expects:
- A pretrained model checkpoint path
- Input directory containing .img OCT volume files
- Output directory path for saving results

Each OCT volume should be 200x1024x200 (B-scan index x depth x lateral position)
"""
import matplotlib.pyplot as plt
import numpy as np

N_CLASSES = 9  # including background

# Zeiss Cirrus Optic Disc Cube 200x200 geometry (same assumption as OCT-ONH)
N_BSCANS = 200            # number of B-scans (slow axis)
N_DEPTH = 1024            # pixels per A-scan (axial)
N_WIDTH = 200             # A-scans per B-scan (lateral)
FOV_X_MM = 6.0            # lateral extent of a B-scan
FOV_Y_MM = 6.0            # extent along the B-scan stacking direction
DEPTH_MM = 2.0            # axial extent
AXIAL_MM = DEPTH_MM / N_DEPTH   # ~1.953 um per pixel
BACKGROUND_LABEL = 240    # label given to the background class in the saved volume
VMAX_UM = 300.0           # upper limit of the fixed color scale (um); vmin is 0

import os
import os.path as osp
import torch
import data.seg_transforms as dt
from data.seg_dataset_volume import segListVolume

from main_nyupitt import parse_args
from main_nyupitt import create_directory

from models.nets.OSNet import OSMGUNet
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


def seg_volume(args, test_loader, to_save_path):
    """
    Segment an OCT volume using the pretrained model.

    Args:
        args: Namespace object containing runtime arguments including device and model path
        test_loader: DataLoader providing OCT volume slices
        to_save_path: Path to save the segmentation volume (.npy)

    Returns:
        numpy.ndarray: Segmented volume of shape (200, 1024, 200) with class labels
            (RNFL = 0, background = 240)
    """
    # load the model
    print('Loading test model ...')
    net = OSMGUNet()
    model = torch.nn.DataParallel(net).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded!')
    cudnn.benchmark = True

    # initialised with the background label so that unprocessed slices are not counted as RNFL (label 0)
    output_volume = np.full((N_BSCANS, N_DEPTH, N_WIDTH), BACKGROUND_LABEL, dtype=np.uint8)

    phase = test_loader.dataset.phase
    if phase == "predict":
        # eval_predict(phase, args, test_loader, model, test_result_path)
        model.eval()
        for iter, (image, imt, imn) in enumerate(test_loader):
            with torch.no_grad():
                print(f"Processing {imn} ...")
                image_var = Variable(image).to(args.device)

                # model forward
                # _, _, output_seg = model(image_var)
                output_seg = model(image_var)
                _, pred_seg = torch.max(output_seg, 1)
                pred_out = pred_seg.squeeze().cpu().data.numpy().astype('uint8')

                # batch_time.update(time.time() - end)
                # end = time.time()

                # DEBUG
                # import numpy as np
                # print(np.unique(pred_out))

                pred_out[pred_out == 0] = 240
                pred_out[pred_out == 1] = 0
                pred_out[pred_out == 2] = 30
                pred_out[pred_out == 3] = 60
                pred_out[pred_out == 4] = 90
                pred_out[pred_out == 5] = 120
                pred_out[pred_out == 6] = 150
                pred_out[pred_out == 7] = 180
                pred_out[pred_out == 8] = 210

                output_volume[iter, :, :] = pred_out

        if to_save_path != "":
            np.save(to_save_path, output_volume)
            print('Output is saved here:\n\t', to_save_path)
        else:
            print("Output is successfully generated.")

        return output_volume

def compute_rnfl_thickness_map(svol, axial_mm=AXIAL_MM, normalize=False):
    """
    Compute the RNFL thickness map from a segmented volume.

    Thickness at each (B-scan, A-scan) position = number of RNFL pixels (label 0)
    along the depth axis x axial pixel size (same idea as OCT-ONH).

    Args:
        svol: segmented volume (B-scans x depth x width); RNFL pixels have label 0
        axial_mm: axial pixel size in mm (default: 2 mm / 1024 for the Cirrus Cube)
        normalize: if True, return the map scaled to [0, 1] (relative values only,
            no physical meaning). Default False.

    Returns:
        numpy.ndarray (B-scans x width): RNFL thickness in micrometers (um).
    """
    thickness_mm = np.count_nonzero(svol == 0, axis=1) * axial_mm
    heatmap = thickness_mm * 1000.0  # mm -> um

    if normalize:
        maxv = heatmap.max()
        if maxv > 0:
            heatmap = heatmap / maxv
    return heatmap


def save_thickness_figure(heatmap_um, save_path, vmax=VMAX_UM,
                          fov_x_mm=FOV_X_MM, fov_y_mm=FOV_Y_MM):
    """
    Save a color-coded RNFL thickness map (jet, fixed scale 0..vmax um) with
    physical axes (mm) and a colorbar (um).
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(heatmap_um, cmap='jet', vmin=0, vmax=vmax,
                   extent=[0, fov_x_mm, fov_y_mm, 0], aspect='equal')
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("RNFL thickness map")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RNFL thickness (\u00b5m)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':

    # region CONFIG
    args = parse_args()

    SAVE_SEGMENTED_VOLUMES = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    args.__dict__["device"] = device
    print("DEVICE:", device)

    # path setting

    dataset_name = args.data_dir.split('/')[-1]

    print('#' * 15, "IMPORTANT PATHS", '#' * 15)
    print('data_dir:',args.data_dir)
    print('log_path:',args.log_path)
    print('model_path:',args.model_path)

    assert args.model_path != "", "MODEL_PATH is not defined."
    model_dir_name = args.model_path.split('/')[-3]
    print(f"This model directory name ({model_dir_name}) is used to create a folder inside the result directory.")
    # endregion

    # NOTE: THESE VALUES ARE DATASET DEPENDENT
    PIXEL_MEAN = [0.14162449977018857] * 3
    PIXEL_STD = [0.09798050174952816] * 3
    normalize = dt.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    test_phase = "predict"
    t = [dt.ToTensor(), normalize]

    # Handle both directory and single file inputs
    if os.path.isdir(args.data_dir):
        all_files = os.listdir(args.data_dir)
    else:
        dir_path, filename = os.path.split(args.data_dir)
        all_files = [filename]
        args.data_dir = dir_path

    for file in all_files:
        if file.find('.img') > -1:
            oct_filepath = os.path.join(args.data_dir, file)
            # vol, enface_image = get_volume_and_enface_image(oct_filepath)

            test_dataset = segListVolume(oct_filepath, dt.Compose(t))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                      num_workers=args.workers, pin_memory=False)
            # endregion

            test_result_path = osp.join(args.log_path, dataset_name, test_phase, model_dir_name)
            create_directory(test_result_path)

            if SAVE_SEGMENTED_VOLUMES:
                to_save_path = os.path.join(test_result_path, file)
            else:
                to_save_path = ""
            output = seg_volume(args, test_loader, to_save_path)

            # DEBUG
            # output = np.load(
            #     r"e:/logs/Normal-ONH-000420-2010-04-22-10-53-54-OD.npy\predict\osmgunet_0.001_t1\Normal-ONH-000420-2010-04-22-10-53-54-OD.npy")

            # RNFL thickness map in micrometers (pixel count x axial pixel size)
            heatmap = compute_rnfl_thickness_map(output)

            base = os.path.join(test_result_path, file.split('.')[0] + '-rnfl-thickness')
            np.save(base + '.npy', heatmap)           # thickness values in um
            save_thickness_figure(heatmap, base + '.png')
