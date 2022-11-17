import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.backends.cudnn as cudnn
from torch.nn import SyncBatchNorm
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
import utils
from utils import CONFIG
import networks


class Trainer(object):

    def __init__(self,
                 train_dataloader,
                 test_dataloader,
                 logger):

        cudnn.benchmark = True

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.sumwriter = SummaryWriter(log_dir="tb_log")
        self.model_config = CONFIG.model
        self.train_config = CONFIG.train
        self.log_config = CONFIG.log
        self.loss_dict = {'rec': None,
                          'comp': None,
                          'lap': None, }
        self.test_loss_dict = {'rec': None,
                               'mse': None,
                               'sad': None}

        self.gauss_filter = torch.tensor([[1., 4., 6., 4., 1.],
                                          [4., 16., 24., 16., 4.],
                                          [6., 24., 36., 24., 6.],
                                          [4., 16., 24., 16., 4.],
                                          [1., 4., 6., 4., 1.]]).cuda()
        self.gauss_filter /= 256.
        self.gauss_filter = self.gauss_filter.repeat(1, 1, 1, 1)

        self.build_model() 
        self.resume_step = None
        self.best_loss = {'mse':1e+8, 'sad':1e+8}

        utils.print_network(self.G, CONFIG.version)

    def build_model(self):
        self.G = networks.get_generator()
        self.G.cuda()

        if CONFIG.dist:
            self.logger.info("Using pytorch synced BN")
            self.G = SyncBatchNorm.convert_sync_batchnorm(self.G)

        self.G_optimizer = torch.optim.AdamW(self.G.parameters(),
                                            lr=self.train_config.base_lr,weight_decay=0.05,
                                            betas=[self.train_config.beta1, self.train_config.beta2])

        if CONFIG.dist:
            # SyncBatchNorm only supports DistributedDataParallel with single GPU per process
            self.G = DistributedDataParallel(self.G, device_ids=[CONFIG.local_rank], output_device=CONFIG.local_rank)
        else:
            self.G = nn.DataParallel(self.G)
        self.build_lr_scheduler()

    def build_lr_scheduler(self):
        """Build cosine learning rate scheduler."""
        #num_steps = int(self.train_config.epochs*len(self.train_dataloader))
        #warmup_steps = int(self.train_config.warmup_epoch*len(self.train_dataloader))
        self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,T_max=self.train_config.epochs-self.train_config.warmup_epoch, last_epoch=-1)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.G_optimizer.zero_grad()

    def train(self):
   
        cp_path = '/experiments/latest_model.pth'
        if os.path.exists(cp_path):
            checkpoint = torch.load(cp_path,map_location='cpu')
            #checkpoint = torch.load(self.train_config.resume_checkpoint, map_location='cpu')
            self.G.load_state_dict(checkpoint['state_dict'],strict=True)
            self.G_optimizer.load_state_dict(checkpoint['opt_state_dict'])
            self.G_scheduler.load_state_dict(checkpoint['lr_state_dict'])
            self.resume_epoch = checkpoint['iter']
            start = self.resume_epoch
        else:
            start = 0
            self.resume_epoch = 0

        moving_max_grad = 0
        moving_grad_moment = 0.999

        for epoch in range(self.resume_epoch+1,self.train_config.epochs):
          if epoch <= self.train_config.warmup_epoch:
                cur_G_lr = utils.warmup_lr(self.train_config.base_lr, epoch, self.train_config.warmup_epoch)
                utils.update_lr(cur_G_lr, self.G_optimizer)
          else:
                self.G_scheduler.step()
                cur_G_lr = self.G_scheduler.get_lr()[0]

          for idx, image_dict in enumerate(self.train_dataloader):
            image, alpha, trimap = image_dict['image'],image_dict['alpha'],image_dict['trimap']
            image = image.cuda(non_blocking=True)
            alpha = alpha.cuda(non_blocking=True)
            trimap = trimap.cuda(non_blocking=True)
            fg_norm, bg_norm = image_dict['fg'].cuda(non_blocking=True),image_dict['bg'].cuda(non_blocking=True)

            self.G.train()
            loss = 0


            """===== Forward G ====="""
            pred = self.G(image, trimap)
            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

            weight_os8 = utils.get_unknown_tensor(trimap)
            weight_os8[...] = 1

            weight_os4 = utils.get_unknown_tensor(trimap)
            weight_os1 = utils.get_unknown_tensor(trimap)
            alpha_pred_os4[weight_os4 == 0] = alpha_pred_os8[weight_os4 == 0]
            alpha_pred_os1[weight_os1 == 0] = alpha_pred_os4[weight_os1 == 0]
            if self.train_config.rec_weight > 0:
                self.loss_dict['rec'] = (self.regression_loss(alpha_pred_os1, alpha, loss_type='l1', weight=weight_os1) * 2 + \
                                         self.regression_loss(alpha_pred_os4, alpha, loss_type='l1', weight=weight_os4) * 1 + \
                                         self.regression_loss(alpha_pred_os8, alpha, loss_type='l1', weight=weight_os8) * 1) / 5.0 * self.train_config.rec_weight

            if self.train_config.comp_weight > 0:
                self.loss_dict['comp'] = (self.composition_loss(alpha_pred_os1, fg_norm, bg_norm, image, weight=weight_os1) * 2 + self.composition_loss(alpha_pred_os4, fg_norm, bg_norm, image, weight=weight_os4) * 1 + self.composition_loss(alpha_pred_os8, fg_norm, bg_norm, image, weight=weight_os8) * 1) / 5.0 * self.train_config.comp_weight 
                
            if self.train_config.lap_weight > 0:
                self.loss_dict['lap'] = (self.lap_loss(logit=alpha_pred_os1, target=alpha, gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os1) * 2 + \
                                         self.lap_loss(logit=alpha_pred_os4, target=alpha, gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os4) * 1 + \
                                         self.lap_loss(logit=alpha_pred_os8, target=alpha, gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os8) * 1) / 5.0 * self.train_config.lap_weight

            for loss_key in self.loss_dict.keys():
                if self.loss_dict[loss_key] is not None and loss_key in ['rec', 'comp', 'lap']:
                    loss += self.loss_dict[loss_key]
            #loss = loss.item()
            """===== Back Propagate ====="""
            self.reset_grad()
            loss.backward()

            """===== Clip Large Gradient ====="""
            if self.train_config.clip_grad:
                if moving_max_grad == 0:
                    moving_max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 1e+6)
                    max_grad = moving_max_grad
                else:
                    max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 2 * moving_max_grad)
                    moving_max_grad = moving_max_grad * moving_grad_moment + max_grad * (1 - moving_grad_moment)

            """===== Update Parameters ====="""
            self.G_optimizer.step()

            """===== Write Log ====="""
            # stdout log
            if idx % self.log_config.logging_step == 0:
                self.write_log(loss, idx, epoch, len(self.train_dataloader), image, cur_G_lr)
                self.sumwriter.add_scalars("loss",dict(train_loss=loss.detach().cpu().numpy()), idx+(epoch-1)*len(self.train_dataloader))
                self.sumwriter.add_scalars("learning_rate",dict(lr=cur_G_lr),idx+(epoch-1)*len(self.train_dataloader))
                #self.test(idx, start)
          if epoch>(self.resume_epoch+1) and epoch % self.train_config.save_epoch==0 and CONFIG.local_rank == 0:
                self.logger.info('Saving the trained models from epoch {}...'.format(epoch))
                self.save_model("latest_model", epoch, loss)
                self.save_model("epoch_"+str(epoch), epoch, loss)

          #torch.cuda.empty_cache()
          

            #"""===== TEST ====="""
          if ((epoch % self.train_config.val_epoch) == 0 or epoch == self.train_config.epochs) and epoch > start:
              self.test(epoch, start)

    def test(self, step, start):
        self.G.eval()
        test_loss = 0
        log_info = ""

        self.test_loss_dict['mse'] = 0
        self.test_loss_dict['sad'] = 0
        for loss_key in self.loss_dict.keys():
            if loss_key in self.test_loss_dict and self.loss_dict[loss_key] is not None:
                self.test_loss_dict[loss_key] = 0

        with torch.no_grad():
            for idx, image_dict in enumerate(self.test_dataloader):
                image, alpha, trimap = image_dict['image'], image_dict['alpha'], image_dict['trimap']
                alpha_shape = image_dict['alpha_shape']
                image = image.cuda()
                alpha = alpha.cuda()
                trimap = trimap.cuda()

                pred = self.G(image, trimap)

                alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
                alpha_pred = alpha_pred_os8.clone().detach()
                weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
                alpha_pred[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]
                weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
                alpha_pred[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]

                h, w = alpha_shape
                alpha_pred = alpha_pred[..., :h, :w]

                trimap = trimap[..., :h, :w]

                weight = utils.get_unknown_tensor(trimap)  # get unknown region (trimap)
                # weight[...] = 1                          # get whole region

                # value of MSE/SAD here is different from test.py and matlab version
                self.test_loss_dict['mse'] += self.mse(((alpha_pred * 255.).int()).float() / 255., alpha, weight)
                # self.test_loss_dict['mse'] += self.mse(alpha_pred, alpha, weight)
                self.test_loss_dict['sad'] += self.sad(alpha_pred, alpha, weight)
                if self.train_config.rec_weight > 0:
                    self.test_loss_dict['rec'] += \
                        self.regression_loss(alpha_pred, alpha, weight=weight) * self.train_config.rec_weight

        # reduce losses from GPUs
        if CONFIG.dist:
            self.test_loss_dict = utils.reduce_tensor_dict(self.test_loss_dict, mode='mean')

        """===== Write Log ====="""
        # stdout log
        for loss_key in self.test_loss_dict.keys():
            if self.test_loss_dict[loss_key] is not None:
                self.test_loss_dict[loss_key] /= len(self.test_dataloader)
                # logging
                log_info += loss_key.upper() + ": {:.4f} ".format(self.test_loss_dict[loss_key])

                test_loss += self.test_loss_dict[loss_key]

        self.logger.info("TEST: LOSS: {:.4f} ".format(test_loss) + log_info)
        self.sumwriter.add_scalars("test_loss",dict(test_loss=test_loss),step)
        self.sumwriter.add_scalars("test_mse_loss",dict(test_mse_loss=self.test_loss_dict['mse']),step)
        self.sumwriter.add_scalars("test_sad_loss",dict(test_sad_loss=self.test_loss_dict['sad']),step)
        torch.cuda.empty_cache()

    def write_log(self, loss, step, epoch, lens, image, cur_G_lr):
        log_info= ''

        # reduce losses from GPUs
        if CONFIG.dist:
            self.loss_dict = utils.reduce_tensor_dict(self.loss_dict, mode='mean')
            loss = utils.reduce_tensor(loss)

        # create logging information
        for loss_key in self.loss_dict.keys():
            if self.loss_dict[loss_key] is not None:
                log_info += loss_key.upper() + ": {:.4f}, ".format(self.loss_dict[loss_key])

        self.logger.debug("Image tensor shape: {}.".format(image.shape))
        log_info = "[{}/{}-{}], ".format(step, lens, epoch) + log_info
        log_info += "lr: {:6f}".format(cur_G_lr)
        self.logger.info(log_info)

    def save_model(self, checkpoint_name, iter, loss):
        """Restore the trained generator and discriminator."""
        torch.save({
            'iter': iter,
            'loss': loss,
            'state_dict': self.G.state_dict(),
            'opt_state_dict': self.G_optimizer.state_dict(),
            'lr_state_dict': self.G_scheduler.state_dict()
        }, os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(checkpoint_name)))
        self.logger.info('Saving models in step {} iter... : {}'.format(iter, '{}.pth'.format(checkpoint_name)))

    @staticmethod
    def mse(logit, target, weight):
        # return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        return Trainer.regression_loss(logit, target, loss_type='l2', weight=weight)

    @staticmethod
    def sad(logit, target, weight):
        return F.l1_loss(logit * weight, target * weight, reduction='sum') / 1000

    @staticmethod
    def regression_loss(logit, target, loss_type='l1', weight=None):
        """
        Alpha reconstruction loss
        :param logit:
        :param target:
        :param loss_type: "l1" or "l2"
        :param weight: tensor with shape [N,1,H,W] weights for each pixel
        :return:
        """
        if weight is None:
            if loss_type == 'l1':
                return F.l1_loss(logit, target)
            elif loss_type == 'l2':
                return F.mse_loss(logit, target)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
        else:
            if loss_type == 'l1':
                return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            elif loss_type == 'l2':
                return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))

    @staticmethod
    def composition_loss(alpha, fg, bg, image, weight, loss_type='l1'):
        """
        Alpha composition loss
        """
        merged = fg * alpha + bg * (1 - alpha)
        return Trainer.regression_loss(merged, image, loss_type=loss_type, weight=weight)

    @staticmethod
    def lap_loss(logit, target, gauss_filter, loss_type='l1', weight=None):
        '''
        Based on FBA Matting implementation:
        https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
        '''

        def conv_gauss(x, kernel):
            x = F.pad(x, (2, 2, 2, 2), mode='reflect')
            x = F.conv2d(x, kernel, groups=x.shape[1])
            return x

        def downsample(x):
            return x[:, :, ::2, ::2]

        def upsample(x, kernel):
            N, C, H, W = x.shape
            cc = torch.cat([x, torch.zeros(N, C, H, W).cuda()], dim=3)
            cc = cc.view(N, C, H * 2, W)
            cc = cc.permute(0, 1, 3, 2)
            cc = torch.cat([cc, torch.zeros(N, C, W, H * 2).cuda()], dim=3)
            cc = cc.view(N, C, W * 2, H * 2)
            x_up = cc.permute(0, 1, 3, 2)
            return conv_gauss(x_up, kernel=4 * gauss_filter)

        def lap_pyramid(x, kernel, max_levels=3):
            current = x
            pyr = []
            for level in range(max_levels):
                filtered = conv_gauss(current, kernel)
                down = downsample(filtered)
                up = upsample(down, kernel)
                diff = current - up
                pyr.append(diff)
                current = down
            return pyr

        def weight_pyramid(x, max_levels=3):
            current = x
            pyr = []
            for level in range(max_levels):
                down = downsample(current)
                pyr.append(current)
                current = down
            return pyr

        pyr_logit = lap_pyramid(x=logit, kernel=gauss_filter, max_levels=5)
        pyr_target = lap_pyramid(x=target, kernel=gauss_filter, max_levels=5)
        if weight is not None:
            pyr_weight = weight_pyramid(x=weight, max_levels=5)
            return sum(Trainer.regression_loss(A[0], A[1], loss_type=loss_type, weight=A[2]) * (2 ** i) for i, A in
                       enumerate(zip(pyr_logit, pyr_target, pyr_weight)))
        else:
            return sum(Trainer.regression_loss(A[0], A[1], loss_type=loss_type, weight=None) * (2 ** i) for i, A in
                       enumerate(zip(pyr_logit, pyr_target)))
