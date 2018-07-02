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

# def sample_from_bounded_gaussian(x):
#     return max(-2*x, min(2*x, np.random.randn()*x))

def sample_from_bounded_gaussian(mean, var):
    return max(mean-var+1e-2, min(mean+var, mean+np.random.randn()*var))

class MPII(data.Dataset):
    def __init__(self, jsonfile, img_folder, inp_res=256, out_res=64,
                 std_size=200, is_train=True):

        # self.is_collecting_data = is_collecting_data
        self.is_train = is_train
        self.img_folder = img_folder
        self.inp_res = inp_res
        self.out_res = out_res
        self.std_size = std_size
        self.rotation_means = np.arange(-60, 61, 20)
        self.rotaiton_var = 5
        print 'rotation gaussian number is', len(self.rotation_means)
        print 'rotation means: ', self.rotation_means
        # self.rotation_means = np.arange(-50, 50, 10)
        self.num = len(self.rotation_means)
        # if grnd_scale_tensor is not None:
        #     self.grnd_scale_tensor = grnd_scale_tensor

            # self.index_list = index_list
            # print self.grnd_scale_tensor.size(), type(self.grnd_scale_tensor)

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
        if self.is_train:
            print 'total training images: ', len(self.train)
        else:
            print 'total validation images: ', len(self.valid)

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
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = torch.Tensor(a['joint_self'])
        # pts[:, 0:2] -= 1  # Convert pts to zero based
        pts = pts[:, 0:2]
        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(a['objpos'])
        # print c
        s = torch.Tensor([a['scale_provided']])
        # r = torch.Tensor([1])
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
        # img = Image.open(img_path)
        # SCALE_VAR, ROTATE_VAR = 0.25, 30
        imgs = [None]*(self.num)
        # imgs_flip = [None] * (self.num)
        # imgs[0], _ = self.gen_img_heatmap(np.array([1]), np.array([0]), origin_img, annot)
        heatmaps = [None]*self.num
        pts_augs = [None] * self.num
        # heatmaps_flip = [None] * self.num
        scales = [None]*self.num
        rotations = [None] * self.num
        centers = [None]*self.num
        origin_pts = [None] * self.num
        normalizers = [None] * self.num
        rot_idx = torch.zeros(self.num).long()
        for i in range(0, len(self.rotation_means)):
            rot_idx[i] = i
            scales[i] = s.clone()
            rotations[i] = torch.Tensor([self.rotation_means[i]])
            centers[i] = c.clone()
            origin_pts[i] = pts.clone()
            normalizers[i] = normalizer
            imgs[i], heatmaps[i], pts_augs[i] = self.gen_img_heatmap(c.clone(), s.clone(),
                                                                     self.rotation_means[i],
                                                                     img.clone(), pts.clone())
        # exit()
        return imgs, heatmaps, centers, scales, rotations, \
               origin_pts, normalizers, rot_idx, index, pts_augs

    def __len__(self):
        # return 10
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)

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
        pts_aug = torch.from_numpy(pts_aug).float()

        return inp, heatmap, pts_aug

