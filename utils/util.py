# Xi Peng, Feb 2017
import os, sys
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch

class PoseTrainHistory():
    """store statuses from the 1st to current epoch"""
    def __init__(self):
        self.epoch = []
        self.lr = []
        self.losses = []
        self.pckh = []
        self.best_pckh = 0.
        self.is_best = True

    def update(self, epoch, lr, loss, pckh):
        # lr, epoch, loss, rmse (OrderedDict)
        # epoch = OrderedDict([('epoch',1)] )
        # loss = OrderedDict( [('train_loss',0.1),('val_loss',0.2)] )
        self.epoch.append(epoch)
        self.lr.append(lr)
        self.losses.append(loss)
        self.pckh.append(pckh)

        self.is_best = pckh['val_pckh'] > self.best_pckh
        self.best_pckh = max(pckh['val_pckh'], self.best_pckh)

    def state_dict(self):
        dest = OrderedDict()
        dest['epoch'] = self.epoch
        dest['lr'] = self.lr
        dest['loss'] = self.losses
        dest['pckh'] = self.pckh
        dest['best_pckh'] = self.best_pckh
        dest['is_best'] = self.is_best
        return dest

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.losses = state_dict['loss']
        self.pckh = state_dict['pckh']
        self.best_pckh = state_dict['best_pckh']
        self.is_best = state_dict['is_best']

class ASNTrainHistory():
    """store statuses from the 1st to current epoch"""
    def __init__(self):
        self.epoch = []
        self.lr = []
        self.losses = []
        # self.pckh = []
        self.lowest_loss = 100.
        self.is_best = True

    def update(self, epoch, lr, loss, pckh=None):
        # lr, epoch, loss, rmse (OrderedDict)
        # epoch = OrderedDict([('epoch',1)] )
        # loss = OrderedDict( [('train_loss',0.1),('val_loss',0.2)] )
        self.epoch.append(epoch)
        self.lr.append(lr)
        self.losses.append(loss)
        # self.pckh.append(pckh)

        self.is_best = loss['train_loss'] < self.lowest_loss
        self.lowest_loss = min(loss['train_loss'], self.lowest_loss)

    def state_dict(self):
        dest = OrderedDict()
        dest['epoch'] = self.epoch
        dest['lr'] = self.lr
        dest['loss'] = self.losses
        # dest['pckh'] = self.pckh
        dest['lowest_loss'] = self.lowest_loss
        dest['is_best'] = self.is_best
        return dest

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.losses = state_dict['loss']
        # self.pckh = state_dict['pckh']
        self.lowest_loss = state_dict['lowest_loss']
        self.is_best = state_dict['is_best']

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_lr(opt, optimizer, epoch):

    if epoch < 100:
        for param_group in optimizer.param_groups:
                print(param_group['lr'])
        return
    elif epoch == 100:
        opt.lr = opt.lr * 0.2
    elif epoch == 140:
        opt.lr = opt.lr * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr
        print(param_group['lr'])

def adjust_lr_v2(opt, optimizer, epoch):
    if epoch < 150:
        for param_group in optimizer.param_groups:
                print(param_group['lr'])
        return
    elif epoch == 150:
        opt.lr = opt.lr * 0.2
    elif epoch == 200:
        opt.lr = opt.lr * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr
        print(param_group['lr'])

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def gen_groundtruth(pred_distri, indexes, pckh_regular, pckh_agent):
    # indexes has size of n x m, where n is the sample number
    # and m is the augmentation indexes for a sample,
    # for occlusion, we could multiple occlusion indexes.
    grnd_distri = pred_distri.data.clone()
    # print pred_distri[0], grnd_distri[0]
    # grnd_distri[0, 0] += 1
    # print pred_distri[0], grnd_distri[0]
    # print 'indexes type is', type(indexes)
    # print 'grnd_distri type is', type(grnd_distri)
    # exit()
    assert len(pckh_regular) == len(pckh_agent)
    assert len(pred_distri.size())  == 2
    increase_ratio = 0.2
    decrease_ratio = 0.5
    thres = (1. / pred_distri.size(1)) * 2
    # thres = 0.3
    # print grnd_scale_distri[0]
    # print pred_scale_distri, grnd_scale_distri
    for k in range(0, len(pckh_regular)):
        # print tmp_ditri[k]
        if pckh_regular[k] - pckh_agent[k] > 0:
            increase = 0
            tmp_other_indexes = range(0, len(grnd_distri[k]))
            # print grnd_distri[k], indexes[k]
            # flag = False
            tmp_arr = torch.zeros(indexes.size(1))
            for j in range(0, indexes.size(1)):
                per_increase = increase_ratio * grnd_distri[k, indexes[k, j]]
                # print 'per_increase type is', type(per_increase)
                # upper bound of each probability
                # if grnd_distri[k, indexes[k, j]] + per_increase <= thres:
                # print indexes[k, j], per_increase
                increase += per_increase
                tmp_arr[j] = per_increase
                # print increase
                grnd_distri[k, indexes[k, j]] += per_increase
                # else:
                #     flag = True
                tmp_other_indexes.remove(indexes[k, j])
                # print tmp_other_indexes
            # exit()
            # print grnd_distri[k]
            for t in tmp_other_indexes:
                grnd_distri[k, t] -= increase / len(tmp_other_indexes)
            # print grnd_distri[k]
            # exit()
                # check whether the probability is less than 0 after the deduction
                # if grnd_distri[k, t] < 0:
                #     grnd_distri[k, t] = 0
            # if flag:
            #     print pred_distri[k], increase, tmp_arr, indexes[k], grnd_distri[k], grnd_distri[k].sum()
            #     exit()
            # grnd_distri[k] /= grnd_distri[k].sum()
        elif pckh_regular[k] - pckh_agent[k] <= 0:
            decrease = 0
            tmp_other_indexes = range(0, len(grnd_distri[k]))
            # tmp_arr = torch.zeros(indexes.size(1))
            # print grnd_distri[k], indexes[k]
            for j in range(0, indexes.size(1)):
                per_decrease = decrease_ratio * grnd_distri[k, indexes[k, j]]
                decrease += per_decrease
                # tmp_arr[j] = per_decrease
                # print per_decrease, indexes[k, j]
                # exit()
                grnd_distri[k, indexes[k, j]] -= per_decrease
                tmp_other_indexes.remove(indexes[k, j])
                # print tmp_other_indexes
            # print grnd_distri[k]
            for t in tmp_other_indexes:
                grnd_distri[k, t] += decrease / len(tmp_other_indexes)
            # print pred_distri[k], indexes[k], tmp_arr, decrease, grnd_distri[k], grnd_distri[k].sum()
            # exit()
            # grnd_distri[k] /= grnd_distri[k].sum()
            # print tmp_ditri[k], grnd_scale_distri[0]
            # exit()
        # post proprocessing, ensure no less than 0 and no larger than threshold
        larger_thres_part = 0
        less_zeros_part = 0
        no_larger_thres_indexes = range(0, len(grnd_distri[k]))
        no_less_zero_indexes = range(0, len(grnd_distri[k]))
        for t in range(0, len(grnd_distri[k])):
            if grnd_distri[k, t] > thres:
                tmp_excess = grnd_distri[k, t] - thres
                larger_thres_part += tmp_excess
                grnd_distri[k, t] = thres
                no_larger_thres_indexes.remove(t)
            elif grnd_distri[k, t] < 0:
                less_zeros_part += grnd_distri[k, t]
                grnd_distri[k, t] = 0
                no_less_zero_indexes.remove(t)
        gap = larger_thres_part + less_zeros_part
        if gap > 0:
            avg_gap = gap / len(no_larger_thres_indexes)
            # print larger_thres_part, less_zeros_part, gap, avg_gap
            for t in no_larger_thres_indexes:
                    grnd_distri[k, t] += avg_gap
        elif gap < 0:
            avg_gap = gap / len(no_less_zero_indexes)
            for t in no_less_zero_indexes:
                    grnd_distri[k, t] += avg_gap
                    if grnd_distri[k, t] < 0:
                        grnd_distri[k, t] = 0
        grnd_distri[k] /= grnd_distri[k].sum()

    grnd_distri = torch.autograd.Variable(grnd_distri, requires_grad=False)
    return grnd_distri

def save_drop_count(drop_count, lost_joint_count_path):
    # print(drop_count)
    total_count = 0
    for k in drop_count:
        total_count += drop_count[k]
    # print(total_count)
    zero_pos_neg = {'zero': 0, 'pos': 0, 'neg': 0}
    total_count = float(total_count)
    for k in drop_count:
        drop_count[k] = drop_count[k] / total_count
        if k == 0:
            zero_pos_neg['zero'] += drop_count[k]
        elif k > 0:
            zero_pos_neg['pos'] += drop_count[k]
        else:
            zero_pos_neg['neg'] += drop_count[k]
    # print(drop_count)
    msg = ''
    for k, v in drop_count.items():
        msg += '%d: %.3f\t' % (k, v)
    msg += 'total: %d\t' % total_count
    msg += '\n'
    for k, v in zero_pos_neg.items():
        msg += '%s: %.3f\t' % (k, v)
    msg += '\n'
    print(msg)
    with open(lost_joint_count_path, 'a+') as count_file:
        count_file.write(msg)

