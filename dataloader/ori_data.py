import cv2
import os
import math
import numbers
import random
import logging
import numpy as np
import json
import torch
from   torch.utils.data import Dataset
from   torch.nn import functional as F
from   torchvision import transforms

from   utils import CONFIG

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha = sample['image'][:,:,::-1], sample['alpha']
        
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
     
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        
        # normalize image
        image /= 255.

        if self.phase == "train":
            # convert GBR images to RGB
            merged = sample['merged'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['merged'] = torch.from_numpy(merged).sub_(self.mean).div_(self.std)
            # del sample['image_name']

        sample['image'], sample['alpha'] = \
            torch.from_numpy(image), torch.from_numpy(alpha)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)

        return sample

class Resize(object):
    def __init__(self, scale):
        self.scale = scale
    
    def __call__(self, sample):
        alpha = sample['alpha']
        image = sample['merged'] 
        
        sample['merged'] = cv2.resize(image, self.scale, interpolation=cv2.INTER_CUBIC)
           
        sample['alpha'] = cv2.resize(alpha, self.scale, interpolation=cv2.INTER_CUBIC)

        return sample
 
class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        merged, alpha = sample['merged'], sample['alpha']
        rows, cols, ch = merged.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, merged.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, merged.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        merged = cv2.warpAffine(merged, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['merged'], sample['alpha'] = merged, alpha

        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        sample_ori = sample.copy()
        merged, alpha = sample['merged'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample_ori
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        merged = cv2.cvtColor(merged.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        merged[:, :, 0] = np.remainder(merged[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = merged[:, :, 1][alpha > 0].mean()
        if np.isnan(sat_bar):
            return sample_ori
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = merged[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        merged[:, :, 1] = sat
        # Value noise
        val_bar = merged[:, :, 2][alpha > 0].mean()
        if np.isnan(val_bar):
            return sample_ori
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = merged[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        merged[:, :, 2] = val
        # convert back to BGR space
        merged = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
        sample['merged'] = merged*255

        return sample

class Composite(object):
    def __call__(self, sample):
        merged, alpha = sample['merged'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        merged[merged < 0 ] = 0
        merged[merged > 255] = 255

        sample['image'] = merged
        return sample

class DataGenerator(Dataset):
    def __init__(self, phase="train"):
        self.phase = phase

        if self.phase == "train":
            #data_path = '/apdcephfs/share_1227775/yandali/data/mat_dataset/adobe_composition-1k/Test_set/merged/'
            data_path = '/dockerdata/imagenet/ILSVRC/Data/CLS-LOC/train/'
            self.name_list = []
            for root,dirs,files in os.walk(data_path):
                for f in files:
                    self.name_list.append(os.path.join(root,f)) 

        train_trans = [
            Resize(scale=(224,224)),
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            RandomJitter(),
            Composite(),
            ToTensor(phase="train") ]

        self.transform = {
            'train':
                transforms.Compose(train_trans)
        }[phase]

    def _load_random_bg(self):
        random_len = len(self.name_list)
        bg_idx = random.randint(0,random_len-1)
        bg_path = self.name_list[bg_idx]

        return bg_path

    def crop_bbox_from_img(self, image, param1, param2):
        h,w = image.shape[:2]
        rand_h = random.randint(int(param1*h),int(param2*h))
        rand_w = random.randint(int(param1*w),int(param2*w))
        top = random.randint(0,h-rand_h)
        left = random.randint(0,w-rand_w)
        bottom = top+rand_h
        right = left+rand_w
       
        crop_bbox = (left,top,right,bottom)

        return crop_bbox

    def generate_alpha(self, image, bbox):
        left,top,right,bottom = bbox
        h,w = image.shape[:2]
        alpha = np.random.randn(h,w)
        alpha[top:bottom, left:right] = 1 

        return alpha 


    def __getitem__(self, idx):
        if self.phase == "train":
            image = cv2.imread(self.name_list[idx])
            bg_path = self._load_random_bg()
            bg = cv2.imread(bg_path)
            resize_bg = cv2.resize(bg,(image.shape[1],image.shape[0]),cv2.INTER_CUBIC)
            crop_fg_region = self.crop_bbox_from_img(image,0.5,0.9)
            left,top,right,bottom = crop_fg_region 
            crop_fg = image[top:bottom, left:right]

            fg_region = self.crop_bbox_from_img(crop_fg,0.7,0.9)
            fg_left,fg_top,fg_right,fg_bottom = fg_region
            crop_alpha = self.generate_alpha(crop_fg, fg_region)

            h,w = image.shape[:2]
            alpha = np.zeros(shape=(h,w))

            #crop_h, crop_w = crop_fg.shape[:2]
            alpha[top:bottom,left:right] = crop_alpha
            
            merged = resize_bg
            merge_alpha = crop_alpha[...,None].astype(np.float32) 
            merged[top:bottom,left:right] = resize_bg[top:bottom,left:right]*(1-merge_alpha)+crop_fg*merge_alpha
 
            sample = {'merged':merged, 'alpha': alpha}

        sample = self.transform(sample)

        return sample['image'],sample['alpha']

    def __len__(self):
        if self.phase == "train":
            return len(self.name_list)
        else:
            return len(self.test_data)
