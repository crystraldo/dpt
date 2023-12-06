# DPT: Disentangled Pre-training for Image Matting

### [Paper](https://jianbojiao.com/pdfs/wacv24dpt.pdf) | [Project page](https://crystraldo.github.io/dpt_mat/)
#### Yanda Li, Zilong Huang, Gang Yu, Ling Chen, Yunchao Wei, Jianbo Jiao
#### University of Technology Sydney, Tencent, Beijing Jiaotong University, University of Birmingham
### Accepted by WACV 2024 as an Oral Presentation (2.5%)

![overview](https://crystraldo.github.io/dpt_mat/static/images/teaser.jpg)

---
## Installation

Please refer to requirements.txt for installation.

pip3 install -r requirements.txt

## Get Started
## Training 

### Pre-training with single/multiple GPUs on Imagenet-1k.
You should download the Imagenet-1k dataset and set the train1k_path in config/dpt.toml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 main.py

### Fine-tuning on Composition-1k training set.
You could use MatteFormer for fine-tuning by using our pre-trained DPT model for initialization.

## Testing

### Test for pre-training model
If you want to test your performance during pre-training phase, you could generate test images with the same setting of training phase, and set the test path in config/dpt.toml

It is worth emphasizing that the performance of pre-training is not directly proportional to the performance of fine-tuning

### Test with single GPU on Composition-1k
python3 inference.py

## Model weights for composition-1k based on Matteformer
You can download our model weights from the link:
https://1drv.ms/u/s!AtQiYwqUDNqOjWtwYb0FC1MuTQ0O?e=5BIbCQ
and use inference.py for testing the performance.



