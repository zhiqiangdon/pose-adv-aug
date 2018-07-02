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
from data.pretrain_s_r_agent import MPII
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
logsoftmax = nn.LogSoftmax(dim=1)

def main():
    opt = TrainOptions().parse()
    if opt.sr_dir == '':
        print('sr directory is null.')
        exit()
    sr_pretrain_dir = os.path.join(opt.exp_dir, opt.exp_id,
                                   opt.sr_dir + '-' +
                                   opt.load_prefix_pose[0:-1])
    if not os.path.isdir(sr_pretrain_dir):
        os.makedirs(sr_pretrain_dir)
    train_history = ASNTrainHistory()
    # print(train_history.lr)
    # exit()
    checkpoint_agent = Checkpoint()
    visualizer = Visualizer(opt)
    visualizer.log_path = sr_pretrain_dir + '/' + 'log.txt'
    train_scale_path = sr_pretrain_dir + '/' + 'train_scales.txt'
    train_rotation_path = sr_pretrain_dir + '/' + 'train_rotations.txt'
    val_scale_path = sr_pretrain_dir + '/' + 'val_scales.txt'
    val_rotation_path = sr_pretrain_dir + '/' + 'val_rotations.txt'

    # with open(visualizer.log_path, 'a+') as log_file:
    #     log_file.write(opt.resume_prefix_pose + '.pth.tar\n')
    # lost_joint_count_path = os.path.join(opt.exp_dir, opt.exp_id, opt.astn_dir, 'joint-count.txt')
    # print("=> log saved to path '{}'".format(visualizer.log_path))
    # if opt.dataset == 'mpii':
    #     num_classes = 16
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    print 'collecting training scale and rotation distributions ...\n'
    train_scale_distri = read_grnd_distri_from_txt(train_scale_path)
    train_rotation_distri = read_grnd_distri_from_txt(train_rotation_path)
    dataset = MPII('dataset/mpii-hr-lsp-normalizer.json', '/bigdata1/zt53/data',
                   is_train=True, grnd_scale_distri=train_scale_distri,
                   grnd_rotation_distri=train_rotation_distri)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, shuffle=True,
                                            num_workers=opt.nThreads, pin_memory=True)

    print 'collecting validation scale and rotation distributions ...\n'
    val_scale_distri = read_grnd_distri_from_txt(val_scale_path)
    val_rotation_distri = read_grnd_distri_from_txt(val_rotation_path)
    dataset = MPII('dataset/mpii-hr-lsp-normalizer.json', '/bigdata1/zt53/data',
                   is_train=False, grnd_scale_distri=val_scale_distri,
                   grnd_rotation_distri=val_rotation_distri)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, shuffle=False,
                                             num_workers=opt.nThreads, pin_memory=True)

    agent = model.create_asn(chan_in=256, chan_out=256,
                             scale_num=len(dataset.scale_means),
                             rotation_num=len(dataset.rotation_means), is_aug=True)
    agent = torch.nn.DataParallel(agent).cuda()
    optimizer = torch.optim.RMSprop(agent.parameters(), lr=opt.lr, alpha=0.99,
                                    eps=1e-8, momentum=0, weight_decay=0)
    # optimizer = torch.optim.Adam(agent.parameters(), lr=opt.agent_lr)
    if opt.load_prefix_sr == '':
        checkpoint_agent.save_prefix = sr_pretrain_dir + '/'
    else:
        checkpoint_agent.save_prefix = sr_pretrain_dir + '/' + opt.load_prefix_sr
        checkpoint_agent.load_prefix = checkpoint_agent.save_prefix[0:-1]
        checkpoint_agent.load_checkpoint(agent, optimizer, train_history)
        # adjust_lr(optimizer, opt.lr)
        # lost_joint_count_path = os.path.join(opt.exp_dir, opt.exp_id, opt.asdn_dir, 'joint-count-finetune.txt')
    print 'agent: ', type(optimizer), optimizer.param_groups[0]['lr']

    if opt.dataset == 'mpii':
        num_classes = 16
    hg = model.create_hg(num_stacks=2, num_modules=1,
                         num_classes=num_classes, chan=256)
    hg = torch.nn.DataParallel(hg).cuda()
    if opt.load_prefix_pose == '':
        print('please input the checkpoint name of the pose model')
        exit()
    checkpoint_hg = Checkpoint()
    # checkpoint_hg.save_prefix = os.path.join(opt.exp_dir, opt.exp_id, opt.resume_prefix_pose)
    checkpoint_hg.load_prefix = os.path.join(opt.exp_dir, opt.exp_id,
                                             opt.load_prefix_pose)[0:-1]
    checkpoint_hg.load_checkpoint(hg)

    logger = Logger(sr_pretrain_dir + '/' + 'training-summary.txt',
                    title='training-summary')
    logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss'])

    """training and validation"""
    start_epoch = 0
    if opt.load_prefix_sr != '':
        start_epoch = train_history.epoch[-1]['epoch'] + 1
    for epoch in range(start_epoch, opt.nEpochs):
        # train for one epoch
        train_loss = train(train_loader, hg, agent, optimizer, epoch, visualizer, opt)
        val_loss = validate(val_loader, hg, agent, epoch, visualizer, opt)
        # update training history
        e = OrderedDict([('epoch', epoch)])
        lr = OrderedDict([('lr', optimizer.param_groups[0]['lr'])])
        loss = OrderedDict([('train_loss', train_loss), ('val_loss', val_loss)])
        # pckh = OrderedDict( [('val_pckh', val_pckh)] )
        train_history.update(e, lr, loss)
        # print(train_history.lr[-1]['lr'])
        checkpoint_agent.save_checkpoint(agent, optimizer, train_history, is_asn=True)
        visualizer.plot_train_history(train_history, 'sr')
        logger.append([epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss])
    logger.close()

    # if train_history.is_best:
        #     visualizer.display_imgpts(imgs, pred_pts, 4)


def train(train_loader, hg, agent, optimizer, epoch, visualizer, opt):
    batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    losses_scale = AverageMeter()
    losses_rot = AverageMeter()
    agent.train()
    hg.eval()

    # for i, (img, heatmap, scales, rotations) in enumerate(train_loader):
    for i, (img, scale_distri, rotation_distri, img_index) in enumerate(train_loader):
        # for j in range(0, img_no_s.size(0)):
            # print img_list[j].size()
        #     img_show = imutils.im_to_numpy(img_no_s[j])*255
        #     img_show = Image.fromarray(img_show.astype('uint8'), 'RGB')
        #     img_show.save('debug-images/%d.jpg' % j)
        # exit()
        # , scale_list, rotation_list
        # batch_size = img.size(0)
        """measure data loading time"""
        # print 'batch ', i
        # print img.size(), scale_distri.size(), rotation_distri.size()
        # print img_index
        # print scale_distri
        # print rotation_distri
        # tmp = scale_distri * torch.log(scale_distri)
        # print tmp, tmp.sum(1), tmp.sum() / batch_size
        # exit()
        # print scale_distri, rotation_distri
        scale_distri_var = torch.autograd.Variable(scale_distri.cuda())
        rotation_distri_var = torch.autograd.Variable(rotation_distri.cuda())
        img_var = torch.autograd.Variable(img)
        pred_scale_distri, pred_rotation_distri = hg(img_var, asn=agent, is_half_hg=True, is_aug=True)
        # print 'pred_scale_distri size: ', pred_scale_distri.size()
        # print 'pred_rotation_distri size: ', pred_rotation_distri.size()
        # exit()
        log_pred_scale_distri = logsoftmax(pred_scale_distri)

        # print test_logsoftmax, log_pred_scale_distri
        loss_scale = F.kl_div(log_pred_scale_distri, scale_distri_var) * scale_distri.size(1)
        if loss_scale.data[0] < -1e-8:
            print 'scale loss < 0'
            print 'pred_scale_distri: ', pred_scale_distri.data
            print 'log_pred_scale_distri: ', log_pred_scale_distri.data
            print 'scale_distri: ', scale_distri
            exit()
        log_pred_rotation_distri = logsoftmax(pred_rotation_distri)
        loss_rotation = F.kl_div(log_pred_rotation_distri, rotation_distri_var) * rotation_distri.size(1)
        # print 'rotation loss is ', loss2.data
        loss = loss_scale + loss_rotation

        # exit()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_scale.update(loss_scale.data[0])
        losses_rot.update(loss_rotation.data[0])
        losses.update(loss.data[0])
        loss_dict = OrderedDict([('scale loss', losses_scale.avg),
                                 ('rotation loss', losses_rot.avg),
                                 ('loss', losses.avg)])
        if i % opt.print_freq == 0 or i == len(train_loader) - 1:
            visualizer.print_log(epoch, i, len(train_loader), value1=loss_dict)

    return losses.avg

def validate(val_loader, hg, agent, epoch, visualizer, opt):
    batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    losses_scale = AverageMeter()
    losses_rot = AverageMeter()
    agent.eval()
    hg.eval()

    # for i, (img, heatmap, scales, rotations) in enumerate(train_loader):
    for i, (img, scale_distri, rotation_distri, img_index) in enumerate(val_loader):
        # for j in range(0, img_no_s.size(0)):
        #     # print img_list[j].size()
        #     img_show = imutils.im_to_numpy(img_no_s[j])*255
        #     img_show = Image.fromarray(img_show.astype('uint8'), 'RGB')
        #     img_show.save('debug-images/%d.jpg' % j)
        # exit()
        # , scale_list, rotation_list
        # batch_size = img.size(0)
        scale_distri_var = torch.autograd.Variable(scale_distri.cuda())
        rotation_distri_var = torch.autograd.Variable(rotation_distri.cuda())
        img_var = torch.autograd.Variable(img)
        pred_scale_distri, pred_rotation_distri = hg(img_var, asn=agent, is_half_hg=True, is_aug=True)
        log_pred_scale_distri = logsoftmax(pred_scale_distri)
        # print test_logsoftmax, log_pred_scale_distri
        loss_scale = F.kl_div(log_pred_scale_distri, scale_distri_var) * scale_distri.size(1)
        log_pred_rotation_distri = logsoftmax(pred_rotation_distri)
        loss_rotation = F.kl_div(log_pred_rotation_distri, rotation_distri_var) * rotation_distri.size(1)
        # print 'rotation loss is ', loss2.data
        loss = loss_scale + loss_rotation

        losses_scale.update(loss_scale.data[0])
        losses_rot.update(loss_rotation.data[0])
        losses.update(loss.data[0])
        loss_dict = OrderedDict([('scale loss', losses_scale.avg),
                                 ('rotation loss', losses_rot.avg),
                                 ('loss', losses.avg)])
        if i % opt.print_freq == 0 or i == len(val_loader) - 1:
            visualizer.print_log(epoch, i, len(val_loader), value1=loss_dict)

    return losses.avg


def read_grnd_distri_from_txt(load_path):
    grnd_distri_list = []
    k = 0
    with open(load_path, 'r') as fd:
        for line in fd:
            # print scale_distri_list[k], line, line.split()
            # exit()
            tmp_vec = [torch.FloatTensor([float(x)]) for x in line.split()]
            # print 'before concat: ', tmp_vec
            tmp_vec = torch.cat(tmp_vec)
            # print 'after concat: ', tmp_vec
            assert abs((1 - tmp_vec.sum())) < 0.1
            grnd_distri_list.append(tmp_vec)
            # print tmp_vec, scale_distri_list[k]
            # k += 1
            # if k == 2:
            #     exit()

    return grnd_distri_list

if __name__ == '__main__':
    main()
