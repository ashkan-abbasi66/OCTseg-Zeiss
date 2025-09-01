"""
USAGE:
    srun -u -p gpu --gres gpu:v100:1 --mem-per-cpu=32GB --time=0-2
    python compute_rnfl_thickness_map_batch.py --model-path ./my-pretrained-model/model/model_best.pth.tar --batch-size 1 -date-dir ./ --log_path ./logs-temp/
Output:
    logs/nyu_for_annotation/predict/osmgunet_0.001_t1
"""
import matplotlib.pyplot as plt
import numpy as np

N_CLASSES = 9  # including background

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
    # load the model
    print('Loading test model ...')
    net = OSMGUNet()
    model = torch.nn.DataParallel(net).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded!')
    cudnn.benchmark = True

    output_volume = np.zeros((200, 1024, 200))

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
    args.log_path = f"logs/"

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

            heatmap = np.zeros((200, 200))
            for i in range(200):  # We have 200 slices (b-scans), each with size of 1024x200
                # Count pixels classified as 0 in the current slice
                bscan = output[i, :, :]
                slice_counts = np.count_nonzero(bscan == 0, axis=0)
                heatmap[i, :] = slice_counts
            heatmap /= heatmap.reshape(-1).max()

            plt.imsave(os.path.join(test_result_path, file.split('.')[0] + '-rnfl-thickness.png'),
                       heatmap, cmap='gray')
