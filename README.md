DPT
## Title
cvpr 1730
## Paper id
title: Disentangled Pre-training for Image Matting

## model weights for composition-1k based on Matteformer
You can download our model weights from anonymous link:


## Installation

Please refer to requirements.txt for installation.
pip3 install -r requirements.txt

## Get Started

### Pre_training with single/multiple GPUs on Imagenet-1k
You should download the Imagenet-1k dataset and set the train1k_path in config/dpt.toml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 main.py

If you want to test your performance during pre-training phase, you could generate test images with the same setting of training phase, and set the test path in config/dpt.toml

## Test with single GPU on Composition-1k
python3 inference.py




