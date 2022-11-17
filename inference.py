import os
import cv2
import toml
import argparse
import numpy as np

import torch
from torch.nn import functional as F

import utils
from   utils import CONFIG
import networks


def single_inference(model, image_dict):

    with torch.no_grad(): 
        image, trimap, alpha = image_dict['image'], image_dict['trimap'], image_dict['alpha']
        image = image.cuda()
        trimap = trimap.cuda()
        alpha = alpha.cuda()
        # run model
        pred = model(image, trimap)
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        # refinement
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]

        h, w = image_dict['alpha_shape']
        alpha_pred = alpha_pred[..., :h, :w]
        trimap = trimap[..., :h, :w]
        weight = utils.get_unknown_tensor(trimap)    
        mse = F.mse_loss(alpha_pred*weight, alpha*weight,reduction='sum') / (torch.sum(weight) + 1e-8)
        sad = F.l1_loss(alpha_pred*weight, alpha*weight,  reduction='sum') / 1000
        return alpha_pred, mse,sad


def generator_tensor_dict(image_path, alpha_path, trimap_path):
    # read images
    image = cv2.imread(image_path)
    trimap = cv2.imread(trimap_path, 0)
    alpha = cv2.imread(alpha_path, 0)/255.

    sample = {'image': image, 'alpha': alpha, 'trimap':trimap, 'alpha_shape':alpha.shape}

    # reshape
    h, w = sample["alpha_shape"]
    
    if h % 32 != 0 or w % 32 != 0:
      target_h = 32 * ((h - 1) // 32 + 1)
      target_w = 32 * ((w - 1) // 32 + 1)
      pad_h = target_h - h
      pad_w = target_w - w
      padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
      padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0, pad_w)), mode="reflect")

      sample['image'] = padded_image
      sample['trimap'] = padded_trimap

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    # convert GBR images to RGB
    image, trimap, alpha = sample['image'][:,:,::-1], sample['trimap'], sample['alpha']

    alpha[alpha < 0 ] = 0
    alpha[alpha > 1] = 1
    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)
    alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
    # trimap configuration
    trimap[trimap < 85] = 0
    trimap[trimap >= 170] = 2
    trimap[trimap >= 85] = 1

    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['trimap'], sample['alpha'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long), torch.from_numpy(alpha)
    sample['image'] = sample['image'].sub_(mean).div_(std)
    # trimap to one-hot 3 channel
    sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()

    # add first channel
    sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

    return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/configs/dpt.toml')
    parser.add_argument('--checkpoint', type=str, default='/checkpoints/best_model_mse.pth')


    parser.add_argument('--image-dir', type=str, default='adobe_composition-1k/Test_set/merged/', help="input image dir")
    parser.add_argument('--mask-dir', type=str, default='adobe_composition-1k/Test_set/alpha_copy', help="input trimap dir")
    parser.add_argument('--trimap-dir', type=str, default='adobe_composition-1k/Test_set/trimaps/', help="input trimap dir")

    parser.add_argument('--output', type=str, default='/output/', help="output dir")

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    utils.make_dir(os.path.join(args.output, 'pred_alpha'))

    # build model
    model = networks.get_generator(is_train=False)
    model.cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()
    mse = 0.0
    sad = 0.0
    for i, image_name in enumerate(os.listdir(args.image_dir)):

        # assume image and mask have the same file name
        image_path = os.path.join(args.image_dir, image_name)
        trimap_path = os.path.join(args.trimap_dir, image_name)
        alpha_path = os.path.join(args.mask_dir, image_name)
        image_dict = generator_tensor_dict(image_path, alpha_path, trimap_path)
        alpha_pred, per_mse, per_sad = single_inference(model, image_dict)
        sad+=per_sad
        mse+=per_mse
        # save images
        print('[{}/{}] inference done : {}'.format(i, len(os.listdir(args.image_dir)), os.path.join(args.output, 'pred_alpha', image_name)))
        alpha_pred = alpha_pred.squeeze().cpu().numpy()
        cv2.imwrite(os.path.join(args.output,'pred_alpha', image_name), alpha_pred*255)
    print(mse/1000)
    print(sad/1000)
