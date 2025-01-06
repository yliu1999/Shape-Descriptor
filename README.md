# Shape-Descriptor
Shape Descriptor and Geometric Detail Fusion for Robust Category-Level Object Pose Estimation

## Requirements
**Prepare the environment**

python 3.8

pytorch 1.13.0

CUDA 11.6

**Datasets**

Please Download the following data [NOCS](https://github.com/hughw19/NOCS_CVPR2019).

## Training
To train the model, please run:

<code>python train.py  --gpus 0,1</code>


## Evaluation
To test the model, please run:

<code>python test.py --gpus 0,1 --test_epoch [YOUR EPOCH]</code>

## Model Checkpoints
[model ckeckpoints](https://drive.google.com/drive/folders/194Nuz10NVW2iOj_ccdVc9tqZ8jRwc7kP?usp=sharing)

## Acknowledgement
Our code is developed upon [IST-Net](https://github.com/CVMI-Lab/IST-Net?tab=readme-ov-file).
Our dataset is provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019).













