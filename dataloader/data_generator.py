import cv2
import os
import math
import numbers
import random
import logging
import numpy as np

from PIL import Image
import json
import torch
from   torch.utils.data import Dataset
from   torch.nn import functional as F
from   torchvision import transforms
from skimage import transform
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
        image, alpha = sample['image'], sample['alpha']
        
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
            fg = sample['fg'].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)
            bg = sample['bg'].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)

        sample['image'], sample['alpha'] = \
            torch.from_numpy(image), torch.from_numpy(alpha)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)

        trimap = torch.zeros(sample['alpha'].shape[1:])
        tri_alpha = alpha.squeeze()
        trimap[tri_alpha==0] = 3
        trimap[tri_alpha==1] = 2
        trimap[trimap==0] = 1
        trimap[trimap==3] = 0

        trimap = F.one_hot(trimap.to(torch.long),num_classes=3)
        trimap = trimap.permute(2,0,1).float() 
        sample['trimap'] = trimap
 
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
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

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
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample_ori
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        if np.isnan(sat_bar):
            return sample_ori
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        if np.isnan(val_bar):
            return sample_ori
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample

class ToTensorTest(object):
    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase

    def __call__(self, sample):
        image, alpha, trimap, mask = sample['image'][:,:,::-1], sample['alpha'], sample['trimap'], sample['mask']

        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1

        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 2
        trimap[trimap >= 85] = 1

        mask = np.expand_dims(mask.astype(np.float32), axis=0)
        image /= 255.

        sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap).to(torch.long)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2,0,1).float()

        sample['mask'] = torch.from_numpy(mask).float()

        return sample

class OriginScale(object):
  def __call__(self, sample):
    h, w = sample["alpha_shape"]

    if h % 32 == 0 and w % 32 == 0:
       return sample

    target_h = 32 * ((h - 1) // 32 + 1)
    target_w = 32 * ((w - 1) // 32 + 1)
    pad_h = target_h - h
    pad_w = target_w - w

    padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
    padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0, pad_w)), mode="reflect")
    padded_mask = np.pad(sample['mask'], ((0,pad_h), (0, pad_w)), mode="reflect")

    sample['image'] = padded_image
    sample['trimap'] = padded_trimap
    sample['mask'] = padded_mask

    return sample

class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None]) 
        sample['image'] = image
        return sample

class DataGenerator(Dataset):
    def __init__(self, data_path=None, phase="train"):
        self.phase = phase

        if self.phase == "train":
            #data_path = '/dockerdata/imagenet/ILSVRC/Data/CLS-LOC/train/'
            data_path = data_path 
            self.name_list = []
            for root,dirs,files in os.walk(data_path):
                for f in files:
                    self.name_list.append(os.path.join(root,f)) 
        else:
            self.merged = data.merged
            self.trimap = data.trimap
            self.alpha = data.alpha

        train_trans = [
            #Resize(scale=(224,224)),
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            RandomJitter(),
            Composite(),
            ToTensor(phase="train") ]

        test_trans = [ OriginScale(), ToTensorTest() ]
        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'test':
                transforms.Compose(test_trans),
            'val':
                transforms.Compose([OriginScale(),ToTensorTest()])
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

    def generate_alpha(self, image):
        alpha = np.zeros(shape=(32,32))
        alpha_ff = alpha.flatten()
        lens = len(alpha_ff)
        keep = random.sample(range(0,lens),int(0.25*lens))
        keep1 = keep[:int(0.5*len(keep))]
        keep2 = keep[int(0.5*len(keep)):]
        out = np.random.randint(0,255,size=(224,224))
        alpha_ff[keep1] = -1
        alpha_ff[keep2] = 255
        alpha_fd = alpha_ff.reshape(32,32)
        alpha_resize = transform.resize(alpha_fd,(224,224),order=0)
        out[alpha_resize==-1] = 0
        out[alpha_resize==255] = 255
        out = out/255.

        return out

    def _load_image(self, path):
        try:
            im = Image.open(path)
        except:
            #if not os.path.exists(path):
              # print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, idx):
        if self.phase == "train":
            image = self._load_image(self.name_list[idx]).convert('RGB')
            bg_path = self._load_random_bg()
            bg = self._load_image(bg_path).convert('RGB')
            image = np.array(image)
            bg = np.array(bg)
            image = cv2.resize(image,(224,224),cv2.INTER_CUBIC)
            bg = cv2.resize(bg,(224,224),cv2.INTER_CUBIC)
            alpha = self.generate_alpha(image)
            image_name = self.name_list[idx].split('/')[-1]
 
            sample = {'alpha': alpha, 'fg':image,'bg':bg, 'image_name': image_name}

        else:
            image = cv2.imread(self.merged[idx])
            alpha = cv2.imread(self.alpha[idx], 0)/255.
            trimap = cv2.imread(self.trimap[idx], 0)
            mask = (trimap >= 170).astype(np.float32)
            image_name = os.path.split(self.merged[idx])[-1]

            sample = {'image': image, 'alpha': alpha, 'trimap': trimap, 'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape}
        sample = self.transform(sample)
        return sample

    def __len__(self):
        if self.phase == "train":
            return len(self.name_list)
        else:
            return len(self.trimap)
