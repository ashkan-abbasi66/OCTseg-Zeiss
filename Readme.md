
# OCTseg-Zeiss

**This is a U-Net based method for Retinal layer segmentation from OCT B-scans.** 
- The OCT b-scan images are assumed to be acquired with a **Cirrus HD-OCT (Zeiss)**. 
  - Each OCT volume covers a 6 × 6 × 2 mm³ area of the retina and is stored as a 200 × 200 × 1024 (horizontal × vertical × depth) data cube. Selected B-scans were then extracted to form the dataset.

<p align="center">
<img width="33%" src=./Fig1.png>
</p>


- This package was **tested** with:
  - Python 3.8.16 and PyTorch 1.12.1
  - CUDA 11.3.1 and cuDNN 8.2.0.53-11.3 (for GPU training)

- An **example dataset** is provided [here](./my-dataset-example) to **only show the structure of the dataset folder**. The dataset is proprietery and cannot be shared. To use the code or train it with a different dataset, you should preserve the same directory structure.

- A **pretrained model** is saved [here](./my-pretrained-model). This model can be used for running the model on your own set of OCT b-scan images. See [here](#test-the-pretrained-model) and provide the appropriate path.

---

# USAGE EXAMPLE

## Train the model


**Example command:** 

`python main_nyupitt.py --lr 0.001 --batch-size 1 --epoch 50 --data-dir d:/data/NYU-OCTseg-dataset/onh-3subsets --log_path d:/logs --test-name nyu-segmenter`

The trained model and its test outputs will be saved under `d:/logs/OCTseg-Zeiss/onh-3subsets` in two subdirectories: `train` and `test`.
Each subdirectory will contain an output folder named `nyu-segmenter_0.001`, where the folder name is derived from the `--test-name` and `--lr` arguments.
- Path to the trained model: `d:/logs/OCTseg-Zeiss/onh-3subsets/train/nyu-segmenter_0.001/model`. The best and last models will be saved.
- Path to the test outputs: `d:\logs\OCTseg-Zeiss\onh-3subsets\test\nyu-segmenter_0.001`.


## Test the Pretrained Model

After training, the model will be saved at `d:/logs/OCTseg-Zeiss/onh-3subsets/train/nyu-segmenter_0.001/model`. You can then test it in either of the following two scenarios.


**Note**: The commands below assume you have trained the model as described in the previous section. If you plan to use the provided pretrained model (available [here](./my-pretrained-model)), make sure to set the `--model-path` accordingly.


### With Ground-Truth Segmentation Masks
Test images are stored at `onh-3subsets\test\img` and their corresponding masks are stored at `onh-3subsets\test\mask`. Then, you need to specify the `--model-path`. 

**Example command:**

`python main_nyupitt.py --test-name nyu-segmenter --lr 0.001 --batch-size 1 --data-dir d:/data/NYU-OCTseg-dataset/onh-3subsets --log_path d:/logs --model-path d:/logs/OCTseg-Zeiss/onh-3subsets/train/nyu-segmenter_0.001/model/model_best.pth.tar`

### Without Segmentation Masks

When running a pretrained model on OCT B-scans without corresponding segmentation masks, in addition `--model-path`, you must:
1. Use the `--predict` flag
2. Ensure the data directory contains a `onh-3subsets/predict/img` folder where input images (without ground-truth masks) are stored.

**Example command:**

`python main_nyupitt.py --test-name nyu-segmenter --lr 0.001 --batch-size 1 --data-dir d:/data/NYU-OCTseg-dataset/onh-3subsets --log_path d:/logs --model-path d:/logs/OCTseg-Zeiss/onh-3subsets/train/nyu-segmenter_0.001/model/model_best.pth.tar --predict`

The output results will be saved in `d:/logs/OCTseg-Zeiss/onh-3subsets/predict/nyu-segmenter_0.001`

## Benchmarking the method
Since I did not have access to the exact dataset of the reference 1, a direct comparison is not feasible. However, my results are also based on the same cohort. In the reference 1, the average dice coefficient for the RNFL thickness map and all layes (when their method was trained in a fully supervised manner) were reported to be 0.88 and 0.8, respectively. Here are mine:

| **Average Dice** | RNFL (Dice_1) | GCL+IPL (Dice_2) | INL (Dice_3) | OPL (Dice_4) | ONL (Dice_5) | IS (Dice_6) | OS (Dice_7) | RPE (Dice_8) |
|------------------|---------------|------------------|--------------|--------------|--------------|-------------|-------------|--------------|
| **0.84**         | 0.87          | 0.85             | 0.82         | 0.75         | 0.93         | 0.82        | 0.87        | 0.85         |


# Reference
1. The dataset has been prepared based on this paper:
  Sedai, S., Antony, B., Rai, R., Jones, K., Ishikawa, H., Schuman, J., Wollstein, G., & Garnavi, R. (2019). *Uncertainty guided semi-supervised segmentation of retinal layers in OCT images*. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 282–290). Springer, Cham. [https://doi.org/10.1007/978-3-030-32239-7_32](https://doi.org/10.1007/978-3-030-32239-7_32)
2. The U-Net architecture is adapted from [MGU-Net](https://github.com/Jiaxuan-Li/MGU-Net).
