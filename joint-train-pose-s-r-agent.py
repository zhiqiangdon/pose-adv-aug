# python train-hg-astn-alter-batch.py --exp_id test --resume_prefix_pose lr-0.00025-20-new.pth.tar --resume_prefix_asn lr-0.00025-80.pth.tar --joint_dir joint-astn
# Zhiqiang Tang, May 2017
# python train-hg-adv.py --exp_id test --resume_prefix_pose pose-lr-0.00025-20.pth.tar --resume_prefix_asn asn-lr-0.00025-15.pth.tar
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
from data.joint_train_pose import MPII
from data.joint_train_s_r_agent import AGENT
import models.asn_stacked_hg as model
from utils.util import AverageMeter
from utils.util import PoseTrainHistory
from utils.util import ASNTrainHistory
from utils.visualizer import Visualizer
from utils.checkpoint import Checkpoint
from utils.util import adjust_lr
from pylib import HumanAcc, HumanPts, HumanAug, Evaluation
from utils import util, imutils
from utils.logger import Logger
cudnn.benchmark = True
softmax = nn.Softmax(dim=1)
dataset = AGENT('dataset/mpii-hr-lsp-normalizer.json', '/bigdata1/zt53/data')

def main():
    opt = TrainOptions().parse()
    if opt.joint_dir == '':
        print('joint directory is null.')
        exit()
    joint_dir = os.path.join(opt.exp_dir, opt.exp_id,
                             opt.joint_dir + '-' +
                             opt.load_prefix_pose[0:-1])
    # joint_dir = os.path.join(opt.exp_dir, opt.exp_id,
    #                          opt.joint_dir)
    if not os.path.isdir(joint_dir):
        os.makedirs(joint_dir)

    visualizer = Visualizer(opt)
    visualizer.log_path = joint_dir + '/' + 'train-log.txt'

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    # lost_joint_count_path = os.path.join(opt.exp_dir, opt.exp_id,
    #                                      opt.joint_dir, 'joint-count.txt')
    if opt.dataset == 'mpii':
        num_classes = 16
    hg = model.create_hg(num_stacks=2, num_modules=1,
                         num_classes=num_classes, chan=256)
    hg = torch.nn.DataParallel(hg).cuda()
    """optimizer"""
    optimizer_hg = torch.optim.RMSprop(hg.parameters(), lr=opt.lr, alpha=0.99,
                                    eps=1e-8, momentum=0, weight_decay=0)
    if opt.load_prefix_pose == '':
        print('please input the checkpoint name of the pose model')
        exit()
    train_history_pose = PoseTrainHistory()
    checkpoint_hg = Checkpoint()
    if opt.load_checkpoint:
        checkpoint_hg.load_prefix = joint_dir + '/' + opt.load_prefix_pose[0:-1]
        checkpoint_hg.load_checkpoint(hg, optimizer_hg, train_history_pose)
    else:
        checkpoint_hg.load_prefix = os.path.join(opt.exp_dir, opt.exp_id) + \
                                    '/' + opt.load_prefix_pose[0:-1]
        checkpoint_hg.load_checkpoint(hg, optimizer_hg, train_history_pose)
        for param_group in optimizer_hg.param_groups:
            param_group['lr'] = opt.lr
    checkpoint_hg.save_prefix = joint_dir + '/pose-'
    # trunc_index = checkpoint.save_prefix_pose.index('lr-0.00025-85')
    # checkpoint.save_prefix_pose = checkpoint.save_prefix_pose[0:trunc_index]
    # print(checkpoint.save_prefix_pose)
    print 'hg optimizer: ', type(optimizer_hg), optimizer_hg.param_groups[0]['lr']

    agent_sr = model.create_asn(chan_in=256, chan_out=256,
                                scale_num=len(dataset.scale_means),
                                rotation_num=len(dataset.rotation_means),
                                is_aug=True)
    agent_sr = torch.nn.DataParallel(agent_sr).cuda()
    optimizer_sr = torch.optim.RMSprop(agent_sr.parameters(), lr=opt.agent_lr,
                                       alpha=0.99, eps=1e-8, momentum=0,
                                       weight_decay=0)
    if opt.load_prefix_sr == '':
        print('please input the checkpoint name of the sr agent.')
        exit()
    train_history_sr = ASNTrainHistory()
    checkpoint_sr = Checkpoint()
    if opt.load_checkpoint:
        checkpoint_sr.load_prefix = joint_dir + '/' + opt.load_prefix_sr[0:-1]
        checkpoint_sr.load_checkpoint(agent_sr, optimizer_sr, train_history_sr)
    else:
        sr_pretrain_dir = os.path.join(opt.exp_dir, opt.exp_id,
                                       opt.sr_dir + '-' + opt.load_prefix_pose[0:-1])
        checkpoint_sr.load_prefix = sr_pretrain_dir + '/' + opt.load_prefix_sr[0:-1]
        checkpoint_sr.load_checkpoint(agent_sr, optimizer_sr, train_history_sr)
        for param_group in optimizer_sr.param_groups:
            param_group['lr'] = opt.agent_lr
    checkpoint_sr.save_prefix = joint_dir + '/agent-'
    # trunc_index = checkpoint.save_prefix_asn.index('lr-0.00025-80')
    # checkpoint.save_prefix_asn = checkpoint.save_prefix_asn[0:trunc_index]
    # print(checkpoint.save_prefix_asn)
    # adjust_lr(optimizer_asn, 5e-5)
    print 'agent optimizer: ', type(optimizer_sr), optimizer_sr.param_groups[0]['lr']

    train_dataset_hg = MPII('dataset/mpii-hr-lsp-normalizer.json',
                            '/bigdata1/zt53/data', is_train=True)
    train_loader_hg = torch.utils.data.DataLoader(train_dataset_hg, batch_size=opt.bs,
                                                  shuffle=True, num_workers=opt.nThreads,
                                                  pin_memory=True)
    val_dataset_hg = MPII('dataset/mpii-hr-lsp-normalizer.json',
                          '/bigdata1/zt53/data', is_train=False)
    val_loader_hg = torch.utils.data.DataLoader(val_dataset_hg, batch_size=opt.bs,
                                                shuffle=False, num_workers=opt.nThreads,
                                                pin_memory=True)
    train_dataset_agent = AGENT('dataset/mpii-hr-lsp-normalizer.json',
                                '/bigdata1/zt53/data', separate_s_r=True)
    train_loader_agent = torch.utils.data.DataLoader(train_dataset_agent, batch_size=opt.bs,
                                                     shuffle=True, num_workers=opt.nThreads,
                                                     pin_memory=True)

    # idx = range(0, 16)
    # idx_pckh = [e for e in idx if e not in (6, 7, 8, 9, 12, 13)]
    if not opt.is_train:
        visualizer.log_path = joint_dir + '/' + 'val-log.txt'
        val_loss, val_pckh, predictions = validate(val_loader_hg, hg,
                                                   train_history_pose.epoch[-1]['epoch'],
                                                   visualizer, num_classes)
        checkpoint_hg.save_preds(predictions)
        return
    logger = Logger(joint_dir + '/' + 'pose-training-summary.txt',
                    title='pose-training-summary')
    logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train PCKh', 'Val PCKh'])
    """training and validation"""
    start_epoch_pose = train_history_pose.epoch[-1]['epoch'] + 1
    epoch_sr = train_history_sr.epoch[-1]['epoch'] + 1

    for epoch in range(start_epoch_pose, opt.nEpochs):
        adjust_lr(opt, optimizer_hg, epoch)
        # train hg for one epoch
        train_loss_pose, train_pckh = train_hg(train_loader_hg, hg, optimizer_hg,
                                               agent_sr, epoch, visualizer, opt)
        # util.save_drop_count(drop_count, lost_joint_count_path)
        # evaluate on validation set
        val_loss, val_pckh, predictions = validate(val_loader_hg, hg, epoch,
                                                   visualizer, num_classes)
        # visualizer.display_imgpts(imgs, pred_pts, 4)
        # exit()
        # update training history
        e_pose = OrderedDict( [('epoch', epoch)] )
        lr_pose = OrderedDict( [('lr', optimizer_hg.param_groups[0]['lr'])] )
        loss_pose = OrderedDict( [('train_loss', train_loss_pose),
                                  ('val_loss', val_loss)])
        pckh = OrderedDict( [('train_pckh', train_pckh),
                             ('val_pckh', val_pckh)] )
        train_history_pose.update(e_pose, lr_pose, loss_pose, pckh)
        checkpoint_hg.save_checkpoint(hg, optimizer_hg,
                                      train_history_pose,
                                      predictions)
        visualizer.plot_train_history(train_history_pose)
        logger.append([epoch, optimizer_hg.param_groups[0]['lr'],
                       train_loss_pose, val_loss, train_pckh, val_pckh])
        # exit()
        # if train_history_pose.is_best:
        #     visualizer.display_imgpts(imgs, pred_pts, 4)

        # train agent_sr for one epoch
        train_loss_sr = train_agent_sr(train_loader_agent, hg, agent_sr,
                                       optimizer_sr, epoch_sr, visualizer, opt)
        e_sr = OrderedDict([('epoch', epoch_sr)])
        lr_sr = OrderedDict([('lr', optimizer_sr.param_groups[0]['lr'])])
        loss_sr = OrderedDict([('train_loss', train_loss_sr),
                               ('val_loss', 0)])
        train_history_sr.update(e_sr, lr_sr, loss_sr)
        # print(train_history.lr[-1]['lr'])
        checkpoint_sr.save_checkpoint(agent_sr, optimizer_sr,
                                      train_history_sr,
                                      is_asn=True)
        visualizer.plot_train_history(train_history_sr, 'sr')
        # exit()
        epoch_sr += 1

    logger.close()

def train_hg(train_loader, hg, optimizer_hg,
             agent_sr, epoch, visualizer, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_hg_regular = AverageMeter()
    losses_hg_sr = AverageMeter()
    losses_hg = AverageMeter()
    pckhs_regular = AverageMeter()
    pckhs_sr = AverageMeter()
    pckhs = AverageMeter()

    # switch mode
    hg.train()
    agent_sr.eval()
    # flags = ['neck', 'skip1', 'skip2', 'skip3', 'skip4']
    end = time.time()
    counter = 0
    # idx_pckh = [e for e in idx if e not in (6, 7, 8, 9, 12, 13)]
    drop_count = {}
    is_asn = False
    for i, (img_std, img, heatmap, c, s, r, grnd_pts,
            normalizer, img_index) in enumerate(train_loader):
        """measure data loading time"""
        # print img_index
        data_time.update(time.time() - end)
        # batch_size = img.size(0)
        # save_imgs(img_std, 'std-imgs')
        # save_imgs(img, 'regular-aug-imgs')
        # input and groundtruth
        if i % 2 == 0:
            print 'regular augmentation'
            img_var = torch.autograd.Variable(img)
            # pts = HumanPts.heatmap2pts(heatmap)
            target_var = torch.autograd.Variable(heatmap.cuda(async=True),
                                                 requires_grad=False)
            out_reg = hg(img_var)
            loss_hg_regular = 0
            for per_out in out_reg:
                tmp_loss = (per_out - target_var) ** 2
                loss_hg_regular = loss_hg_regular + tmp_loss.sum() / tmp_loss.numel()

            optimizer_hg.zero_grad()
            loss_hg_regular.backward()
            optimizer_hg.step()

            losses_hg_regular.update(loss_hg_regular.data[0])
            losses_hg.update(loss_hg_regular.data[0])
            pckh = Evaluation.accuracy_origin_res(out_reg[-1].data.cpu(), c, s,
                                                  [64, 64], grnd_pts, normalizer, r)
            pckhs_regular.update(pckh[0])
            pckhs.update(pckh[0])
        else:
            print 'agent augmentation'
            img_var = torch.autograd.Variable(img_std)

            pred_scale_distri, pred_rotation_distri = hg(img_var, agent_sr,
                                                         is_half_hg=True, is_aug=True)
            pred_scale_distri = softmax(pred_scale_distri)
            pred_rotation_distri = softmax(pred_rotation_distri)
            pred_scale_distri_numpy = pred_scale_distri.data.cpu().numpy()
            pred_rotation_distri_numpy = pred_rotation_distri.data.cpu().numpy()
            # print pred_scale_distri_numpy
            # print pred_rotation_distri_numpy
            # exit()
            scale_index_list = []
            rotation_index_list = []
            for j in range(0, pred_scale_distri_numpy.shape[0]):
                # print len(dataset.scale_means), pred_scale_distri_numpy[j]
                tmp_scale_index = np.random.choice(len(dataset.scale_means), 1,
                                                   p=pred_scale_distri_numpy[j])[0]
                # print pred_scale_distri_numpy[j], np.sum(pred_scale_distri_numpy[j]), tmp_scale_index
                tmp_rotation_index = np.random.choice(len(dataset.rotation_means), 1,
                                                      p=pred_rotation_distri_numpy[j])[0]
                # print pred_rotation_distri_numpy[j], np.sum(pred_rotation_distri_numpy[j]), tmp_rotation_index

                scale_index_list.append(tmp_scale_index)
                rotation_index_list.append(tmp_rotation_index)
                # if j == 1:
                #     exit()
            # exit()
            img, heatmap, c, s, r,\
            grnd_pts, normalizer = load_batch_data(scale_index_list,
                                                   rotation_index_list,
                                                   img_index, opt,
                                                   separate_s_r=False)
            # save_imgs(img, 'agent-aug-imgs')
            # exit()
            img_var = torch.autograd.Variable(img)
            target_var = torch.autograd.Variable(heatmap.cuda(async=True),
                                                 requires_grad=False)
            out_reg = hg(img_var)
            loss_hg_sr = 0
            for per_out in out_reg:
                tmp_loss = (per_out - target_var) ** 2
                loss_hg_sr = loss_hg_sr + tmp_loss.sum() / tmp_loss.numel()

            optimizer_hg.zero_grad()
            loss_hg_sr.backward()
            optimizer_hg.step()
            losses_hg_sr.update(loss_hg_sr.data[0])
            losses_hg.update(loss_hg_sr.data[0])
            pckh = Evaluation.accuracy_origin_res(out_reg[-1].data.cpu(), c, s,
                                                  [64, 64], grnd_pts, normalizer, r)
            pckhs_sr.update(pckh[0])
            pckhs.update(pckh[0])

        loss_dict = OrderedDict([('loss_hg_regular', losses_hg_regular.avg),
                                 ('loss_hg_sr', losses_hg_sr.avg),
                                 ('loss_hg', losses_hg.avg),
                                 ('pckhs_regular', pckhs_regular.avg),
                                 ('pckhs_sr', pckhs_sr.avg),
                                 ('pckh', pckhs.avg)])
        # else:
        #     loss_dict = OrderedDict([('loss_hg_normal', losses_normal_hg.avg),
        #                              ('pckh', pckhs.avg)])
        if i % opt.print_freq == 0 or i == len(train_loader) - 1:
            visualizer.print_log(epoch, i, len(train_loader), value1=loss_dict)
        # if i == 1:
        #     break

    return losses_hg.avg, pckhs.avg

def train_agent_sr(train_loader, hg, agent_sr, optimizer_sr,
                   epoch_sr, visualizer, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_agent_sr = AverageMeter()
    # switch mode
    hg.eval()
    agent_sr.train()
    # flags = ['neck', 'skip1', 'skip2', 'skip3', 'skip4']
    end = time.time()
    for i, (img_std, img_list, heatmap_list, c_list, s_list, r_list,
            grnd_pts_list, normalizer_list, img_index) in enumerate(train_loader):

        # save_imgs(img_std, 'std-imgs')
        # save_imgs(img_list[0], 'regular-s-imgs')
        # save_imgs(img_list[1], 'regular-r-imgs')
        assert len(img_list) == 2
        regular_pckh_list = compute_separated_s_r_pckh(hg, img_list, c_list,
                                                       s_list, r_list,
                                                       grnd_pts_list,
                                                       heatmap_list,
                                                       normalizer_list)

        img_var = torch.autograd.Variable(img_std)
        # output and loss
        pred_scale_distri, pred_rotation_distri = hg(img_var, agent_sr,
                                                     is_half_hg=True, is_aug=True)
        pred_scale_distri = softmax(pred_scale_distri)
        pred_rotation_distri = softmax(pred_rotation_distri)
        pred_scale_distri_numpy = pred_scale_distri.data.cpu().numpy()
        pred_rotation_distri_numpy = pred_rotation_distri.data.cpu().numpy()
        # print pred_scale_distri_numpy
        # print pred_rotation_distri_numpy
        # exit()
        scale_index_list = []
        rotation_index_list = []
        for j in range(0, img_std.size(0)):
            # print len(dataset.scale_means), pred_scale_distri_numpy[j]
            tmp_scale_index = np.random.choice(len(dataset.scale_means), 1,
                                               p=pred_scale_distri_numpy[j])[0]
            # print pred_scale_distri_numpy[j], np.sum(pred_scale_distri_numpy[j]), tmp_scale_index
            tmp_rotation_index = np.random.choice(len(dataset.rotation_means), 1,
                                                  p=pred_rotation_distri_numpy[j])[0]
            # print pred_rotation_distri_numpy[j], np.sum(pred_rotation_distri_numpy[j]), tmp_rotation_index

            scale_index_list.append(tmp_scale_index)
            rotation_index_list.append(tmp_rotation_index)

        img_list, heatmap_list, c_list, s_list,\
        r_list, grnd_pts_list, normalizer_list = load_batch_data(scale_index_list,
                                                                 rotation_index_list,
                                                                 img_index, opt,
                                                                 separate_s_r=True)
        # save_imgs(img_list[0], 'agent-s-imgs')
        # save_imgs(img_list[1], 'agent-r-imgs')
        # exit()
        sr_pckh_list = compute_separated_s_r_pckh(hg, img_list, c_list, s_list, r_list,
                                                  grnd_pts_list, heatmap_list,
                                                  normalizer_list)

        indexes_scale = torch.LongTensor(scale_index_list)
        indexes_scale.unsqueeze_(1)
        # print indexes.size()
        grnd_scale_distri = util.gen_groundtruth(pred_distri=pred_scale_distri,
                                                 indexes=indexes_scale,
                                                 pckh_regular=regular_pckh_list[0],
                                                 pckh_agent=sr_pckh_list[0])
        indexes_rotation = torch.LongTensor(rotation_index_list)
        indexes_rotation.unsqueeze_(1)
        grnd_rotation_distri = util.gen_groundtruth(pred_distri=pred_rotation_distri,
                                                    indexes=indexes_rotation,
                                                    pckh_regular=regular_pckh_list[1],
                                                    pckh_agent=sr_pckh_list[1])
        # print scale_index_list
        # print pred_scale_distri.data.cpu().numpy()
        # print grnd_scale_distri.data.cpu().numpy()
        # print rotation_index_list
        # print pred_rotation_distri.data.cpu().numpy()
        # print grnd_rotation_distri.data.cpu().numpy()
        # exit()
        # grnd_scale_distri = torch.autograd.Variable(grnd_scale_distri, requires_grad=False)
        # grnd_rotation_distri = torch.autograd.Variable(grnd_rotation_distri, requires_grad=False)
        log_pred_scale_distri = torch.log(pred_scale_distri + 1e-7)
        log_pred_rotation_distri = torch.log(pred_rotation_distri + 1e-7)
        loss_scale = F.kl_div(log_pred_scale_distri, grnd_scale_distri) \
                     * grnd_scale_distri.size(1)
        # print loss_scale
        # exit()
        loss_rotation = F.kl_div(log_pred_rotation_distri, grnd_rotation_distri) \
                        * grnd_rotation_distri.size(1)
        loss_agent_sr = loss_scale + loss_rotation
        optimizer_sr.zero_grad()
        loss_agent_sr.backward()
        optimizer_sr.step()
        losses_agent_sr.update(loss_agent_sr.data[0])

        loss_dict = OrderedDict([('loss_agent_sr', losses_agent_sr.avg)])
        # else:
        #     loss_dict = OrderedDict([('loss_hg_normal', losses_normal_hg.avg),
        #                              ('pckh', pckhs.avg)])
        if i % opt.print_freq == 0 or i == len(train_loader) - 1:
            visualizer.print_log(epoch_sr, i, len(train_loader), value1=loss_dict)
        # if i == 1:
        #     break

        return losses_agent_sr.avg


def load_batch_data(scale_index_list, rotation_index_list,
                    img_index, opt, separate_s_r=False):
    dataset.scale_index_list = scale_index_list
    dataset.rotation_index_list = rotation_index_list
    dataset.img_index_list = img_index.numpy().tolist()
    dataset.separate_s_r = separate_s_r
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=img_index.size(0),
                                               shuffle=False, num_workers=opt.nThreads,
                                               pin_memory=True)
    img = None
    heatmap = None
    c = None
    s = None
    r = None
    grnd_pts = None
    normalizer = None
    img_index_copy = img_index.clone()
    # img_index = None

    for j, (img, heatmap, c, s, r, grnd_pts, normalizer,
            img_index) in enumerate(train_loader):

        assert j == 0
        assert (img_index_copy - img_index).sum() == 0

    return img, heatmap, c, s, r, grnd_pts, normalizer

def compute_separated_s_r_pckh(hg, img_list, c_list, s_list, r_list,
                               grnd_pts_list, grnd_heatmap_list,
                               normalizer_list):
    assert len(img_list) == 2
    pckh_list = []
    for k in range(0, 2):
        img_var = torch.autograd.Variable(img_list[k])
        out_reg = hg(img_var)
        tmp_pckh = Evaluation.per_person_pckh(out_reg[-1].data.cpu(),
                                              grnd_heatmap_list[k],
                                              c_list[k], s_list[k],
                                              [64, 64], grnd_pts_list[k],
                                              normalizer_list[k], r_list[k])
        pckh_list.append(tmp_pckh)

    return pckh_list

def save_imgs(imgs, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for j in range(0, imgs.size(0)):
        # print img_list[j].size()
        img_show = imutils.im_to_numpy(imgs[j])*255
        img_show = Image.fromarray(img_show.astype('uint8'), 'RGB')
        img_show.save(save_dir+'/%d.jpg' % j)



def validate(val_loader, net, epoch, visualizer, num_classes):
    batch_time = AverageMeter()
    losses_det = AverageMeter()
    losses = AverageMeter()
    pckhs = AverageMeter()
    pckhs_origin_res = AverageMeter()
    img_batch_list = []
    pts_batch_list = []
    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    net.eval()

    # end = time.time()
    for i, (img, heatmap, center, scale, rot, grnd_pts,
            normalizer, index) in enumerate(val_loader):
        # input and groundtruth
        input_var = torch.autograd.Variable(img, volatile=True)

        heatmap = heatmap.cuda(async=True)
        target_var = torch.autograd.Variable(heatmap)

        # output and loss
        # output1, output2 = net(input_var)
        # loss = (output1 - target_var) ** 2 + (output2 - target_var) ** 2
        output1 = net(input_var)
        loss = 0
        for per_out in output1:
            tmp_loss = (per_out - target_var) ** 2
            loss = loss + tmp_loss.sum() / tmp_loss.numel()

        # flipping the image
        img_flip = img.numpy()[:, :, :, ::-1].copy()
        img_flip = torch.from_numpy(img_flip)
        input_var = torch.autograd.Variable(img_flip, volatile=True)
        # output11, output22 = net(input_var)
        output11 = net(input_var)
        output11 = HumanAug.flip_channels(output11[-1].cpu().data)
        output11 = HumanAug.shuffle_channels_for_horizontal_flipping(output11)
        output = (output1[-1].cpu().data + output11) / 2
        #

        # pckh = Evaluation.accuracy(output, target_var.data.cpu(), idx)
        # pckhs.update(pckh[0])
        pckh_origin_res = Evaluation.accuracy_origin_res(output, center, scale, [64, 64],
                                                         grnd_pts, normalizer, rot)
        pckhs_origin_res.update(pckh_origin_res[0])
        # # print log
        losses.update(loss.data[0])
        loss_dict = OrderedDict( [('loss', losses.avg),
                                  ('pckh', pckhs_origin_res.avg)] )
        visualizer.print_log( epoch, i, len(val_loader), value1=loss_dict)
        # img_batch_list.append(img)
        # pts_batch_list.append(pred_pts*4.)
        # generate predictions
        preds = Evaluation.final_preds(output, center, scale, [64, 64], rot)
        for n in range(output.size(0)):
            predictions[index[n], :, :] = preds[n, :, :]
    # return losses.avg, pckhs.avg, img_batch_list, pts_batch_list
    #     if i == 1:
    #         break
    return losses.avg, pckhs_origin_res.avg, predictions

    # exit()
if __name__ == '__main__':
    main()
