# python train-stacked-residual-aug-pretrain.py --exp_id test --resume_prefix_pose stacked-8-lr-0.00025-10.pth.tar
# Zhiqiang Tang, May 2017
# import sys, warnings, traceback, torch
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#     traceback.print_stack(sys._getframe(2))
# warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True

import os, time
from PIL import Image, ImageDraw
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parameter import Parameter

from options.train_options import TrainOptions
from data.collect_rotation_distri import MPII
import models.asn_stacked_hg as model
from utils.util import AverageMeter
from utils.util import ASNTrainHistory
from utils.visualizer import Visualizer
from utils.checkpoint import Checkpoint
from utils.util import adjust_lr
from utils import imutils
from pylib import HumanAcc, HumanPts, HumanAug, Evaluation
from utils.logger import Logger
cudnn.benchmark = True
idx = torch.LongTensor([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15])

def main():
    opt = TrainOptions().parse()
    if opt.sr_dir == '':
        print('sr directory is null.')
        exit()
    sr_pretrain_dir = os.path.join(opt.exp_dir, opt.exp_id,
                                   opt.sr_dir+'-'+opt.load_prefix_pose[0:-1])
    if not os.path.isdir(sr_pretrain_dir):
        os.makedirs(sr_pretrain_dir)
    train_history = ASNTrainHistory()
    # print(train_history.lr)
    # exit()
    checkpoint_hg = Checkpoint()
    # visualizer = Visualizer(opt)
    # log_name = opt.resume_prefix_pose + 'log.txt'
    # visualizer.log_path = sr_pretrain_dir + '/' + log_name
    train_distri_path = sr_pretrain_dir + '/' + 'train_rotations.txt'
    train_distri_path_2 = sr_pretrain_dir + '/' + 'train_rotations_copy.txt'
    # train_distri_path = sr_pretrain_dir + '/' + 'train_rotations.txt'
    # train_distri_path_2 = sr_pretrain_dir + '/' + 'train_rotations_copy.txt'
    val_distri_path = sr_pretrain_dir + '/' + 'val_rotations.txt'
    val_distri_path_2 = sr_pretrain_dir + '/' + 'val_rotations_copy.txt'
    # val_distri_path = sr_pretrain_dir + '/' + 'val_rotations.txt'
    # val_distri_path_2 = sr_pretrain_dir + '/' + 'val_rotations_copy.txt'

    if opt.dataset == 'mpii':
        num_classes = 16
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    hg = model.create_hg(num_stacks=2, num_modules=1,
                         num_classes=num_classes, chan=256)
    hg = torch.nn.DataParallel(hg).cuda()
    if opt.load_prefix_pose == '':
        print('please input the checkpoint name of the pose model')
        # exit()
    # checkpoint_hg.save_prefix = os.path.join(opt.exp_dir, opt.exp_id, opt.resume_prefix_pose)
    checkpoint_hg.load_prefix = os.path.join(opt.exp_dir, opt.exp_id,
                                             opt.load_prefix_pose)[0:-1]
    checkpoint_hg.load_checkpoint(hg)

    print 'collecting training distributions ...\n'
    train_distri_list = collect_train_valid_data(train_distri_path,
                                                 train_distri_path_2, hg, opt, is_train=True)

    print 'collecting validation distributions ...\n'
    val_distri_list = collect_train_valid_data(val_distri_path,
                                                val_distri_path_2, hg, opt, is_train=False)


def compute_class_ratio(grnd_scale_tensor):
    count = torch.zeros(7)
    print 'list length: ', len(grnd_scale_tensor)
    for i in range(0, len(grnd_scale_tensor)):
        count[grnd_scale_tensor[i]] += 1
    print 'count: ', count
    ratio = count / count.sum()
    print 'ratio: ', ratio
    exit()

def collect_train_valid_data(save_path, save_path_2, hg, opt, is_train):
    dataset = MPII('dataset/mpii-hr-lsp-normalizer.json',
                   '/bigdata1/zt53/data',
                   is_train=is_train)
    # very important to keep shuffle=False
    collect_data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs,
                    shuffle=False, num_workers=opt.nThreads, pin_memory=True)
    # compare_flip_vs_nonflip(collect_data_loader, dataset, hg)
    # exit()
    if os.path.exists(save_path):
        grnd_distri_list = read_grnd_distri_from_txt(save_path)
        # grnd_rotation_tensor = read_grnd_scale_from_txt(rotation_path)
    else:
        grnd_distri_list = \
            collect_data(collect_data_loader, dataset, hg, save_path)
        # grnd_scale_tensor = torch.stack(grnd_distri_list, dim=0)
        # grnd_rotation_tensor = torch.stack(grnd_rotation_list, dim=0)
        print 'saving scale list ...'
        np.savetxt(save_path_2, torch.stack(grnd_distri_list, dim=0).numpy(), fmt='%.2f')

    return grnd_distri_list


def collect_data(data_loader, dataset, hg, save_path):
    rot_num = len(dataset.rotation_means)
    # rotation_num = len(dataset.rotation_means)
    print 'rot_num: ', rot_num
    # print 'rotation_num: ', rotation_num
    # scale_1_idx = scale_num / 2
    # print 'scale_1_idx', scale_1_idx
    grnd_distri_list = []
    # index_list = []
    counter = 0
    hg.eval()
    # multipliers = np.zeros(7)
    # scale_factors = np.arange(0.7, 1.4, 0.1)
    # for i, s in enumerate(scale_factors):
    #     multipliers[i] = s
    # print 'multipliers: ', multipliers
    # exit()
    for i, (img_list, heatmap_list, center_list, scale_list, rot_list,
            grnd_pts_list, normalizer_list, rot_idx, img_index, pts_aug_list)\
            in enumerate(data_loader):
        print '%d/%d' % (i, len(data_loader))
        # pts_aug_back = Evaluation.transform_preds(pts_aug_list[0][0]+1, center_list[0][0],
        #                         scale_list[0][0], [64, 64], rot_list[0][0])
        # print 'grnd_pts: ', grnd_pts_list[0][0]
        # print 'pts_aug_back: ', pts_aug_back
        # exit()
        # print 'rot_idx size ', rot_idx.size()
        # exit()
        # print img_index
        # for j in range(0, len(img_list)):
        #     # print img_list[j].size()
        #     img_show = imutils.im_to_numpy(img_list[j][0])*255
        #     img_show = Image.fromarray(img_show.astype('uint8'), 'RGB')
        #     img_show.save('debug-images/%d.jpg' % j)
        # exit()
        # index_list.append(img_index)
        # print img_index
        # print(len(img_list))
        # print(img_list[0].size())
        # print img_list[0][0, 0, 0, 0], img_list[1][0, 0, 0, 0]
        # print img_list[0][1, 0, 0, 0], img_list[1][1, 0, 0, 0]
        # exit()
        val_img_index = torch.arange(counter, counter+len(img_index)).long()
        assert ((val_img_index-img_index).sum() == 0)
        counter += len(img_index)
        # print scale_ind.size()
        # print scale_ind
        # exit()
        # print rotation_ind.size()
        # print rotation_ind
        # exit()
        unique_num = rot_idx.size(0)
        assert rot_idx.size(1) == rot_num
        # print 'unique_num: ', unique_num
        img = torch.cat(img_list, dim=0)
        # print 'img size: ', img.size()
        # print img[0, 0, 0, 0], img[2, 0, 0, 0]
        # print img[1, 0, 0, 0], img[3, 0, 0, 0]
        # exit()
        pts_aug = torch.cat(pts_aug_list, dim=0)
        heatmap = torch.cat(heatmap_list, dim=0)
        # print 'heatmap size: ', heatmap.size()
        center = torch.cat(center_list, dim=0)
        # print 'center size: ', center.size()
        # print 'center: ', center
        # print 'center list 0: ', center_list[0]
        # exit()
        scale = torch.cat(scale_list, dim=0)
        # print 'scale size: ', scale.size()
        rotation = torch.cat(rot_list, dim=0)
        grnd_pts = torch.cat(grnd_pts_list, dim=0)
        # print 'grnd_pts size: ', grnd_pts.size()
        normalizer = torch.cat(normalizer_list, dim=0)
        # print 'normalizer size: ', normalizer.size()
        # exit()
        # batch_size = img.size(0)
        # print 'batch_size: ', batch_size
        # output and loss
        img_var = torch.autograd.Variable(img, volatile=True)
        out_reg = hg(img_var)
        # loss = 0
        # for per_out in out_reg:
        #     # print 'hg', counter
        #     # counter += 1
        #     # print per_out.size()
        #     per_out = per_out.data.cpu()
        #     loss = loss + (per_out - heatmap) ** 2
        #     # loss = loss + tmp_loss.sum() / tmp_loss.numel()
        # # exit()
        # # print 'loss type: ', type(loss)
        # # print 'loss size: ', loss.size()
        #
        # elm_num = loss.numel() / batch_size
        # # print 'elm_num: ', elm_num
        # loss = loss.view(loss.size(0), -1).sum(1).div_(elm_num)
        # loss = loss.squeeze().numpy()
        # print 'loss: ', loss
        pckhs = Evaluation.per_person_pckh(
            out_reg[-1].data.cpu(), heatmap, center, scale, [64, 64],
            grnd_pts, normalizer, rotation)
        lost_pckhs = 1 - pckhs
        # pckhs = pckhs.numpy()
        # print 'pckhs: ', pckhs
        # print 'pred_counts shape: ', pred_counts.shape
        for j in range(0, unique_num):
            tmp_pckhs = lost_pckhs[j::unique_num]
            # tmp_loss = loss[j::unique_num]
            # print 'tmp_loss:', tmp_loss
            # print 'weighted tmp_loss:', tmp_loss * multipliers
            # print 'tmp_pckh:', tmp_pckhs
            # exit()
            assert (tmp_pckhs.size(0) == rot_num)
            if tmp_pckhs.sum() == 0:
                print 'tmp_pckh: ', tmp_pckhs
                print 'sum of tmp_pckh are zero. Setting equal probabilities ...'
                tmp_distri = torch.ones(tmp_pckhs.size(0)) / tmp_pckhs.size(0)
            elif (tmp_pckhs < 0).any():
                print 'tmp_pckh: ', tmp_pckhs
                print 'some of tmp_pckh is negative. error...'
                exit()
            else:
                tmp_distri = tmp_pckhs.clone()
                tmp_distri = tmp_distri / tmp_distri.sum()
            # print 'tmp_distri: ', tmp_distri
            grnd_distri_list.append(tmp_distri)
            with open(save_path, 'a+') as log_file:
                tmp_distri = tmp_distri.numpy()
                np.savetxt(log_file, tmp_distri.reshape(1, tmp_distri.shape[0]), fmt='%.2f')
            # assert grnd_scale < scale_num
            # grnd_scale = torch.LongTensor([grnd_scale])
            # grnd_scale_list.append(grnd_scale)

        # if i == 0:
        #     break

    return grnd_distri_list

def read_grnd_distri_from_txt(scale_path):

    grnd_distri_list = []
    # k = 0
    with open(scale_path, 'r') as fd:
        for line in fd:
            # print scale_distri_list[k], line, line.split()
            # exit()
            tmp_vec = [torch.FloatTensor([float(x)]) for x in line.split()]
            # print tmp_vec
            # exit()
            tmp_vec = torch.cat(tmp_vec)
            assert torch.abs((1 - tmp_vec.sum())) < 0.1
            grnd_distri_list.append(tmp_vec)
            # print tmp_vec, scale_distri_list[k]
            # print (tmp_vec - scale_distri_list[k]).sum()
            # k += 1
            # if k == 2:
            #     exit()

    return grnd_distri_list

if __name__ == '__main__':
    main()
