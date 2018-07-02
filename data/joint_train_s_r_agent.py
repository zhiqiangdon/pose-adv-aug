# Zhiqiang Tang, May 2017
import os, sys
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from utils import imutils
from pylib import HumanPts, HumanAug

def sample_from_large_gaussian(x):
    return max(-2*x, min(2*x, np.random.randn()*x))

def sample_from_small_gaussian(mean, var):
    return max(mean-var+1e-3, min(mean+var, mean+np.random.randn()*var))

class AGENT(data.Dataset):
    def __init__(self, jsonfile, img_folder, inp_res=256,
                 out_res=64, std_size=200, separate_s_r=False):

        self.separate_s_r = separate_s_r
        self.img_index_list = None
        self.scale_index_list = None
        self.rotation_index_list = None

        self.img_folder = img_folder
        self.inp_res = inp_res
        self.out_res = out_res
        self.std_size = std_size
        self.scale_factor = 0.25
        self.rot_factor = 30
        self.scale_means = np.arange(-0.6, 0.61, 0.2)
        self.scale_var = 0.05
        self.rotation_means = np.arange(-60, 61, 20)
        self.rotaiton_var = 5
        print 'scale gaussian number is', len(self.scale_means)
        print 'rotation gaussian number is', len(self.rotation_means)

        # create train/val split
        with open(jsonfile, 'r') as anno_file:
            self.anno = json.load(anno_file)
        print 'loading json file is done...'
        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['dataset'] == 'MPII':
                if val['objpos'][0] <= 0 or val['objpos'][1] <= 0:
                    print 'invalid center: ', val['objpos']
                    print 'image name: ', val['img_paths']
                    print 'dataset: ', val['dataset']
                    # continue
                if val['isValidation'] == True:
                    self.valid.append(idx)
                else:
                    self.train.append(idx)
        # self.mean, self.std = self._compute_mean()

        # if grnd_scale_tensor is not None:
        #     if self.is_train:
        #         assert len(self.train) == grnd_scale_tensor.size(0)
        #     else:
        #         assert len(self.valid) == grnd_scale_tensor.size(0)

    def _compute_mean(self):
        meanstd_file = 'dataset/mpii_for_mpii_mean.pth.tar'
        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train:
                a = self.anno[index]
                img_path = os.path.join(self.img_folder, a['img_paths'])
                img = imutils.load_image(img_path)  # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
            }
            torch.save(meanstd, meanstd_file)
        print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
        print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']


    def color_normalize(self, x, mean, std):
        if x.size(0) == 1:
            x = x.repeat(3, x.size(1), x.size(2))

        for t, m, s in zip(x, mean, std):
            t.sub_(m).div_(s)
        return x

    def __getitem__(self, index):
        # print 'loading image', index
        if self.img_index_list is None:
            a = self.anno[self.train[index]]
        else:
            idx = self.img_index_list[index]
            a = self.anno[self.train[idx]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = torch.Tensor(a['joint_self'])
        # pts[:, 0:2] -= 1  # Convert pts to zero based
        pts = pts[:, 0:2]
        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(a['objpos'])
        # print c
        s = torch.Tensor([a['scale_provided']])
        # r = torch.FloatTensor([0])
        # exit()
        if a['dataset'] == 'MPII':
            c[1] = c[1] + 15 * s[0]
            s = s * 1.25
            normalizer = a['normalizer'] * 0.6
        elif a['dataset'] == 'LEEDS':
            print 'using lsp data'
            s = s * 1.4375
            normalizer = torch.dist(pts[2, :], pts[13, :])
        else:
            print 'no such dataset {}'.format(a['dataset'])

        # For single-person pose estimation with a centered/scaled figure
        img = imutils.load_image(img_path)
        if self.img_index_list is None:
            s_aug = s * (2 ** (sample_from_large_gaussian(self.scale_factor)))
            r_aug = sample_from_large_gaussian(self.rot_factor)
            if np.random.uniform(0, 1, 1) <= 0.6:
                r_aug = np.array([0])
        else:
            gaussian_mean_scale = self.scale_means[self.scale_index_list[index]]
            scale_factor = sample_from_small_gaussian(gaussian_mean_scale, self.scale_var)
            gaussian_mean_rotation = self.rotation_means[self.rotation_index_list[index]]
            r_aug = sample_from_small_gaussian(gaussian_mean_rotation, self.rotaiton_var)
            s_aug = s * (2 ** scale_factor)
        if self.separate_s_r:
            img_list = [None]*2
            heatmap_list = [None]*2
            c_list = [c.clone()]*2
            s_list = [s_aug.clone(), s.clone()]
            r_list = [torch.FloatTensor([0]), torch.FloatTensor([r_aug])]
            grnd_pts_list = [pts.clone(), pts.clone()]
            # print 'type of normalizaer: ', type(normalizer)
            normalizer_list = [normalizer, normalizer]
            img_list[0], heatmap_list[0] = self.gen_img_heatmap(c.clone(), s_aug.clone(), 0,
                                                                img.clone(), pts.clone())
            img_list[1], heatmap_list[1] = self.gen_img_heatmap(c.clone(), s.clone(), r_aug,
                                                                img.clone(), pts.clone())
            if self.img_index_list is not None:
                return img_list, heatmap_list, c_list, s_list,\
                       r_list, grnd_pts_list, normalizer_list, idx
            else:
                inp_std, _ = self.gen_img_heatmap(c.clone(), s.clone(), 0,
                                                  img.clone(), pts.clone())
                return inp_std, img_list, heatmap_list, c_list, s_list, r_list,\
                       grnd_pts_list, normalizer_list, index
        else:
            if np.random.random() <= 0.5:
                img = torch.from_numpy(HumanAug.fliplr(img.numpy())).float()
                pts = HumanAug.shufflelr(pts, width=img.size(2), dataset='mpii')
                c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(np.random.uniform(0.6, 1.4)).clamp_(0, 1)
            img[1, :, :].mul_(np.random.uniform(0.6, 1.4)).clamp_(0, 1)
            img[2, :, :].mul_(np.random.uniform(0.6, 1.4)).clamp_(0, 1)

            inp, heatmap = self.gen_img_heatmap(c.clone(), s_aug.clone(), r_aug,
                                                img.clone(), pts.clone())
            # if self.separate_s_r is false, then self.img_index_list is not None
            # so return idx instead of index
            r_aug = torch.FloatTensor([r_aug])
            return inp, heatmap, c, s_aug, r_aug, pts, normalizer, idx

    def __len__(self):
        if self.img_index_list is None:
            return len(self.train)
        else:
            return len(self.img_index_list)

    def gen_img_heatmap(self, c, s, r, img, pts):
        # Prepare image and groundtruth map
        # print s[0]/s0[0], r
        inp = HumanAug.crop(imutils.im_to_numpy(img), c.numpy(),
                            s.numpy(), r, self.inp_res, self.std_size)
        inp = imutils.im_to_torch(inp).float()
        # inp = self.color_normalize(inp, self.mean, self.std)
        pts_aug = HumanAug.TransformPts(pts.numpy(), c.numpy(),
                                        s.numpy(), r, self.out_res, self.std_size)

        idx_indicator = (pts[:, 0] <= 0) | (pts[:, 1] <= 0)
        idx = torch.arange(0, pts.size(0)).long()
        idx = idx[idx_indicator]
        pts_aug[idx, :] = 0
        # Generate ground truth
        heatmap, pts_aug = HumanPts.pts2heatmap(pts_aug, [self.out_res, self.out_res], sigma=1)
        heatmap = torch.from_numpy(heatmap).float()
        # pts_aug = torch.from_numpy(pts_aug).float()

        return inp, heatmap

