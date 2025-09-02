
# OCTseg-Zeiss

**This is a U-Net–based method for retinal layer segmentation from OCT B-scans, including code for training, testing, a pretrained model, and RNFL thickness map computation.**

- The OCT b-scan images are assumed to be acquired with a **Cirrus HD-OCT (Zeiss)**. 
  - Each OCT volume covers a 6 × 6 × 2 mm³ area of the retina and is stored as a 200 × 200 × 1024 (horizontal × vertical × depth) data cube. Selected B-scans were then extracted to form the dataset.

<p align="center">
<img width="33%" src=./Fig1.png>
</p>


- This package was **tested** with:
  - Python 3.8.16 and PyTorch 1.12.1
  - CUDA 11.3.1 and cuDNN 8.2.0.53-11.3 (for GPU training)

---

# Train the Model


An **example dataset** is provided [here](./my-dataset-example) to only **illustrate the required folder structure**. The original dataset is proprietary and cannot be shared. To train the model on your own data, you must follow the same directory structure. 


**Training command:** 

```commandline
python main_nyupitt.py --lr 0.001 --batch-size 1 --epoch 50 --data-dir d:/data/NYU-OCTseg-dataset/onh-3subsets --log_path d:/logs --test-name nyu-segmenter
```

After running this command, the trained model and test outputs will be saved in `d:/logs/OCTseg-Zeiss/onh-3subsets` under two subdirectories: `train` and `test`.
Each subdirectory contains an output folder named `nyu-segmenter_0.001`, where the folder name is derived from the `--test-name` and `--lr` arguments.
- The trained model (best and last checkpoints) will be stored at `d:/logs/OCTseg-Zeiss/onh-3subsets/train/nyu-segmenter_0.001/model`.
- Test outputs will be saved at `d:\logs\OCTseg-Zeiss\onh-3subsets\test\nyu-segmenter_0.001`.


# Test the Pretrained Model

You can use the pretrained model in two ways. From the training section, the model is saved at: `d:/logs/OCTseg-Zeiss/onh-3subsets/train/nyu-segmenter_0.001/model`. However, in the following examples, we instead reference the pretrained model provided in this repository ([./my-pretrained-model](./my-pretrained-model)) to set the `--model-path`.

## With Ground-Truth Segmentation Masks
Assuming the test images are stored in `onh-3subsets\test\img` and their segmentation masks in `onh-3subsets\test\mask`, inference can be executed using the command below. 

**Test command:**

```commandline
python main_nyupitt.py --test-name nyu-segmenter --lr 0.001 --batch-size 1 --data-dir d:/data/NYU-OCTseg-dataset/onh-3subsets --log_path d:/logs --model-path ./my-pretrained-model/model/model_best.pth.tar
```

## Without Segmentation Masks

When running a pretrained model on OCT B-scans without corresponding segmentation masks, in addition `--model-path`, you must:
1. Use the `--predict` flag
2. Ensure the data directory contains a `onh-3subsets/predict/img` folder where input images (without ground-truth masks) are stored.

**Test command:**

```commandline
python main_nyupitt.py --test-name nyu-segmenter --lr 0.001 --batch-size 1 --data-dir d:/data/NYU-OCTseg-dataset/onh-3subsets --log_path d:/logs --model-path ./my-pretrained-model/model/model_best.pth.tar --predict
```

The output results will be saved in `d:/logs/OCTseg-Zeiss/onh-3subsets/predict/nyu-segmenter_0.001`


# Benchmarking the model

Since the exact dataset from Reference 1 was not available, a direct comparison is not possible. My results, however, are based on the same cohort. In Table 1 of Reference 1, the average Dice scores have been reported as **0.80** across all layers and **0.88** for the RNFL layer in the fully supervised setting. My results are as follows:

| **Average Dice** | RNFL (Dice_1) | GCL+IPL (Dice_2) | INL (Dice_3) | OPL (Dice_4) | ONL (Dice_5) | IS (Dice_6) | OS (Dice_7) | RPE (Dice_8) |
|------------------|---------------|------------------|--------------|--------------|--------------|-------------|-------------|--------------|
| **0.84**         | **0.87**          | 0.85             | 0.82         | 0.75         | 0.93         | 0.82        | 0.87        | 0.85         |



# Application to RNFL Thickness Map Computation

<p align="center">
<img width="33%" src=./Fig2.png>
</p>

The segmentation model can be applied to compute retinal nerve fiber layer thickness maps (RNFLTs) from a folder of OCT volumes.


**Requirements**:
- A pretrained model checkpoint (e.g., `./my-pretrained-model/model/model_best.pth.tar`)
- A dataset directory containing `.img` OCT volume files. An example is saved in `./my-dataset-example/` folder.


**RNFLT computation command:**:

```commandline
python compute_rnfl_thickness_map_batch.py --model-path ./my-pretrained-model/model/model_best.pth.tar --batch-size 1 --data-dir ./my-dataset-example/ --log_path ./logs-temp/
```

This command performs slice-by-slice segmentation of each OCT volume:
- The resulting segmentation volume is optionally saved as a `.npy` file.
- A corresponding RNFL thickness map is computed and saved as a `.png` file.
- All outputs are stored in the directory specified by `--log_path`.


**Notes**: 
- **Optic disc exclusio**n: RNFLT may be better visualized if optic disc is excluded. For simplicity, this script does not consider optic disc exclusion.
- **Colormap choice**: Here, a grayscale colormap is used for visualization. Commercial devices usually employ a different colormap (e.g., `jet` like colormap) for better visual contrast.


# Citation and References

If you find this repository helpful, please consider **starring it** or **citing our work**.

The following references were utilized in the development of this repository:

1. The dataset has been prepared based on this paper:
  Sedai, S., Antony, B., Rai, R., Jones, K., Ishikawa, H., Schuman, J., Wollstein, G., & Garnavi, R. (2019). *Uncertainty guided semi-supervised segmentation of retinal layers in OCT images*. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 282–290). Springer, Cham. [https://doi.org/10.1007/978-3-030-32239-7_32](https://doi.org/10.1007/978-3-030-32239-7_32)
2. The U-Net architecture is adapted from [MGU-Net](https://github.com/Jiaxuan-Li/MGU-Net).
