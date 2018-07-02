# Zhiqiang Tang, Apr 2017
import os, sys
import numpy as np
from PIL import Image
import torch

def approx_PCKh(pred, target, idxs, res):
    # pred: b x n x 2 tensor
    # target: b x n x 2 tensor
    assert(pred.size()==target.size())
    target = target.float()
    # distances between prediction and groundtruth coordinates
    dists = torch.zeros((pred.size(1), pred.size(0)))
    normalize = res/10
    for i in range(pred.size(1)):
        for j in range(pred.size(0)):
            if target[j][i][0] > 0 and target[j][i][1] > 0:
                dists[i][j] = torch.dist(target[j][i], pred[j][i]) / normalize
            else:
                dists[i][j] = -1
    # accuracies based on the distances
    threshold = 0.5
    avg_acc = 0
    bad_idx_count = 0
    for i in range(len(idxs)):
        per_joint_dists = dists[idxs[i]]
        if torch.ne(per_joint_dists, -1).sum() > 0:
            valid_count = per_joint_dists.le(threshold).eq(per_joint_dists.ne(-1)).sum()
            all_count = per_joint_dists.ne(-1).sum()
            # print(valid_count)
            # print(type(valid_count))
            # exit()
            per_joint_acc = float(valid_count) / float(all_count)
            # print(per_joint_dists.le(threshold).eq(per_joint_dists.ne(-1)).sum())
            # print('joint {0} accuracy is {1}' .format(idxs[i]+1, per_joint_acc))
        else:
            per_joint_acc = -1
        if per_joint_acc >= 0:
            avg_acc += per_joint_acc
        else:
            bad_idx_count += 1
    avg_acc = avg_acc / (len(idxs)-bad_idx_count)
    # exit()
    return avg_acc

def approx_PCKh_per(pred, target, idxs, res):
    # pred: b x n x 2 tensor
    # target: b x n x 2 tensor
    assert(pred.size()==target.size())
    target = target.float()
    # distances between prediction and groundtruth coordinates
    dists = torch.zeros((pred.size(1), pred.size(0)))
    normalize = res/10
    for i in range(pred.size(1)):
        for j in range(pred.size(0)):
            if target[j][i][0] > 0 and target[j][i][1] > 0:
                dists[i][j] = torch.dist(target[j][i], pred[j][i]) / normalize
            else:
                dists[i][j] = -1
    # accuracies based on the distances
    threshold = 0.5
    avg_acc = 0
    pckhs = torch.zeros(len(idxs))
    bad_idx_count = 0
    for i in range(len(idxs)):
        per_joint_dists = dists[idxs[i]]
        if torch.ne(per_joint_dists, -1).sum() > 0:
            valid_count = per_joint_dists.le(threshold).eq(per_joint_dists.ne(-1)).sum()
            all_count = per_joint_dists.ne(-1).sum()
            # print(valid_count)
            # print(type(valid_count))
            # exit()
            pckhs[i] = float(valid_count) / float(all_count)
            # print(per_joint_dists.le(threshold).eq(per_joint_dists.ne(-1)).sum())
            # print('joint {0} accuracy is {1}' .format(idxs[i]+1, per_joint_acc))
        else:
            pckhs[i] = -1
        if pckhs[i] >= 0:
            avg_acc += pckhs[i]
        else:
            bad_idx_count += 1
    avg_acc = avg_acc / (len(idxs)-bad_idx_count)
    # exit()
    return avg_acc, pckhs

def PCKh(pred, target, normalizer):
    # pred: m x n x 2 tensor, m is the number of imgs and n is the pts number
    # target: m x n x 2 tensor
    threshold = 0.5
    pts_num = target.size(1)
    img_num = target.size(0)
    assert (pred.size() == target.size())
    # distances between prediction and groundtruth coordinates
    dists = torch.zeros((pts_num, img_num))
    # print 'dists size is ', dists.size()
    for i in range(0, pts_num):
        for j in range(0, img_num):
            if target[j][i][0] > 0 and target[j][i][1] > 0:
                dists[i][j] = torch.dist(target[j][i], pred[j][i]) / normalizer[j]
            else:
                dists[i][j] = -1
    avg_pckh = 0
    pckhs = torch.zeros(pts_num)
    bad_idx_count = 0
    for i in range(0, pts_num):
        per_joint_dists = dists[i]
        if torch.ne(per_joint_dists, -1).sum() > 0:
            valid_count = per_joint_dists.le(threshold).eq(per_joint_dists.ne(-1)).sum()
            all_count = per_joint_dists.ne(-1).sum()
            # print(valid_count)
            # print('hahah')
            # # print(type(valid_count))
            # exit()
            pckhs[i] = float(valid_count) / float(all_count)
            # print(per_joint_dists.le(threshold).eq(per_joint_dists.ne(-1)).sum())
            # print('joint {0} accuracy is {1}' .format(idxs[i]+1, per_joint_acc))
        else:
            pckhs[i] = -1
        if pckhs[i] >= 0:
            avg_pckh += pckhs[i]
        else:
            bad_idx_count += 1
    # exit()
    avg_pckh = avg_pckh / (pts_num - bad_idx_count)
    part_names = ('Head', 'Knee', 'Ankle', 'Shoulder', 'Elbow', 'Wrist', 'Hip')
    part_idxs = np.array([[8,9], [1,4], [0,5], [12,13], [11,14], [10,15], [2,3]])
    # print(part_idxs[0, 0])
    # pckhs_ordered = torch.zeros(len(part_names))
    for i in range(0, len(part_names)):
        print('%s: %.4f' % (part_names[i], (pckhs[part_idxs[i, 0]] + pckhs[part_idxs[i, 1]])/2))
        # print(part_names[i], ': ', (pckhs[part_idxs[i, 0]] + pckhs[part_idxs[i, 1]])/2)
        # pckhs_ordered[i] = (pckhs[part_idxs[i, 0]] + pckhs[part_idxs[i, 1]])/2
        # if pckhs_ordered[i] < 0:
        #     a = 1

    print('Average PCKh is: %.4f' % avg_pckh)
    # return pckhs

def approx_PCKh_samples(pred, target, res):
    # pred: b x n x 2 tensor
    # target: b x n x 2 tensor
    assert(pred.size()==target.size())
    sample_num = pred.size(0)
    target = target.float()
    # distances between prediction and groundtruth coordinates
    dists = torch.zeros((pred.size(1), pred.size(0)))
    normalize = res/10
    for i in range(pred.size(1)):
        for j in range(pred.size(0)):
            if target[j][i][0] > 0 and target[j][i][1] > 0:
                dists[i][j] = torch.dist(target[j][i], pred[j][i]) / normalize
            else:
                dists[i][j] = -1
    # accuracies based on the distances
    threshold = 0.5
    # accuracies = torch.zeros(sample_num)
    counts = torch.zeros(sample_num)
    bad_idx_count = 0
    for i in range(0, sample_num):
        # per_person_dists = dists[idxs, i]
        per_person_dists = dists[:, i]
        if torch.ne(per_person_dists, -1).sum() > 0:
            valid_count = per_person_dists.le(threshold).eq(per_person_dists.ne(-1)).sum()
            # all_count = per_person_dists.ne(-1).sum()
            # print(valid_count)
            # print(type(valid_count))
            # exit()
            # accuracies[i] = float(valid_count) / float(all_count)
            counts[i] = valid_count
            # print(per_joint_dists.le(threshold).eq(per_joint_dists.ne(-1)).sum())
            # print('joint {0} accuracy is {1}' .format(idxs[i]+1, per_joint_acc))
        else:
            # accuracies[i] = 0
            counts[i] = 0
    # exit()
    # return accuracies
    return counts

def correct_predicted_joints(pred, target, res):
    # pred: b x n x 2 tensor
    # target: b x n x 2 tensor
    assert(pred.size()==target.size())
    sample_num = pred.size(0)
    joint_num = pred.size(1)
    target = target.float()
    # distances between prediction and groundtruth coordinates
    dists = torch.zeros((joint_num, sample_num))
    normalize = res/10
    for i in range(pred.size(1)):
        for j in range(pred.size(0)):
            if target[j][i][0] > 0 and target[j][i][1] > 0:
                dists[i][j] = torch.dist(target[j][i], pred[j][i]) / normalize
            else:
                dists[i][j] = -1
    # accuracies based on the distances
    threshold = 0.5
    # accuracies = torch.zeros(sample_num)
    correct_masks = torch.ByteTensor(sample_num, joint_num).zero_()
    # joint_indexes = torch.arange(0, pred.size(1))
    for i in range(0, sample_num):
        # per_person_dists = dists[idxs, i]
        per_person_dists = dists[:, i]
        if torch.ne(per_person_dists, -1).sum() > 0:
            correct_masks[i] = per_person_dists.le(threshold).eq(per_person_dists.ne(-1))
            # all_count = per_person_dists.ne(-1).sum()
            # print(valid_count)
            # print(type(valid_count))
            # exit()
            # accuracies[i] = float(valid_count) / float(all_count)
            # correct_joints.append(joint_indexes[one_correct_msk])
            # print(per_joint_dists.le(threshold).eq(per_joint_dists.ne(-1)).sum())
            # print('joint {0} accuracy is {1}' .format(idxs[i]+1, per_joint_acc))
        # else:
            # accuracies[i] = 0
            # correct_joints.append([])
    # exit()
    # return accuracies
    return correct_masks

def correct_predicted_joints_original_resolution(pred, target, normalizer):
    # pred: b x n x 2 tensor
    # target: b x n x 2 tensor
    assert(pred.size()==target.size())
    sample_num = pred.size(0)
    joint_num = pred.size(1)
    target = target.float()
    # distances between prediction and groundtruth coordinates
    dists = torch.zeros((joint_num, sample_num))
    # normalize = res/10
    for i in range(pred.size(1)):
        for j in range(pred.size(0)):
            if target[j][i][0] > 0 and target[j][i][1] > 0:
                dists[i][j] = torch.dist(target[j][i], pred[j][i]) / normalizer
            else:
                dists[i][j] = -1
    # accuracies based on the distances
    threshold = 0.5
    # accuracies = torch.zeros(sample_num)
    correct_masks = torch.ByteTensor(sample_num, joint_num).zero_()
    # joint_indexes = torch.arange(0, pred.size(1))
    for i in range(0, sample_num):
        # per_person_dists = dists[idxs, i]
        per_person_dists = dists[:, i]
        if torch.ne(per_person_dists, -1).sum() > 0:
            correct_masks[i] = per_person_dists.le(threshold).eq(per_person_dists.ne(-1))
            # all_count = per_person_dists.ne(-1).sum()
            # print(valid_count)
            # print(type(valid_count))
            # exit()
            # accuracies[i] = float(valid_count) / float(all_count)
            # correct_joints.append(joint_indexes[one_correct_msk])
            # print(per_joint_dists.le(threshold).eq(per_joint_dists.ne(-1)).sum())
            # print('joint {0} accuracy is {1}' .format(idxs[i]+1, per_joint_acc))
        # else:
            # accuracies[i] = 0
            # correct_joints.append([])
    # exit()
    # return accuracies
    return correct_masks

def predicted_joints_dist_to_grnd(pred, target, res):
    # pred: b x n x 2 tensor
    # target: b x n x 2 tensor
    assert(pred.size()==target.size())
    sample_num = pred.size(0)
    joint_num = pred.size(1)
    target = target.float()
    # distances between prediction and groundtruth coordinates
    dists = torch.zeros((joint_num, sample_num))
    normalize = res/10
    for i in range(pred.size(1)):
        for j in range(pred.size(0)):
            if target[j][i][0] > 0 and target[j][i][1] > 0:
                dists[i][j] = torch.dist(target[j][i], pred[j][i]) / normalize
            else:
                dists[i][j] = -1
    # accuracies based on the distances
    # threshold = 0.5
    # accuracies = torch.zeros(sample_num)
    dists_to_grnd = torch.zeros(sample_num).zero_()
    # joint_indexes = torch.arange(0, pred.size(1))
    for i in range(0, sample_num):
        # per_person_dists = dists[idxs, i]
        per_person_dists = dists[:, i]
        # print(per_person_dists)
        valid_per_person_joint_mask = per_person_dists.ne(-1)
        # print(valid_per_person_joint_mask)
        valid_per_person_joint_num = valid_per_person_joint_mask.sum()
        # print(valid_per_person_joint_num)
        assert(torch.ne(per_person_dists, -1).sum() == valid_per_person_joint_num)
        if valid_per_person_joint_num > 0:
            valid_per_person_joint_dists = per_person_dists[valid_per_person_joint_mask]
            # print(valid_per_person_joint_dists)
            dists_to_grnd[i] = valid_per_person_joint_dists.sum() / valid_per_person_joint_num
            # all_count = per_person_dists.ne(-1).sum()
            # print(dists_to_grnd)
            # print(type(valid_count))
        # exit()
            # accuracies[i] = float(valid_count) / float(all_count)
            # correct_joints.append(joint_indexes[one_correct_msk])
            # print(per_joint_dists.le(threshold).eq(per_joint_dists.ne(-1)).sum())
            # print('joint {0} accuracy is {1}' .format(idxs[i]+1, per_joint_acc))
        # else:
            # accuracies[i] = 0
            # correct_joints.append([])
    # exit()
    # return accuracies
    return dists_to_grnd