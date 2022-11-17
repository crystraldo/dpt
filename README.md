## DPT
## Title
title: Disentangled Pre-training for Image Matting
## Paper id
cvpr 1730
## model weights for composition-1k based on Matteformer
You can download our model weights from anonymous link:
https://1drv.ms/u/s!AtQiYwqUDNqOjWtwYb0FC1MuTQ0O?e=5BIbCQ

## Installation

Please refer to requirements.txt for installation.
pip3 install -r requirements.txt

## Get Started
## Training 

### Pre-training with single/multiple GPUs on Imagenet-1k.
You should download the Imagenet-1k dataset and set the train1k_path in config/dpt.toml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 main.py

### Fine-tuning on Composition-1k training set.
You could use MatteFormer for fine-tuning by using our pretrained DPT model for initialization.

## Testing

### Test for pre-training model
If you want to test your performance during pre-training phase, you could generate test images with the same setting of training phase, and set the test path in config/dpt.toml

## Test with single GPU on Composition-1k
python3 inference.py




