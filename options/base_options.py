# Xi Peng, May 2017
import argparse
import os
from utils import util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_dir', type=str, default='./dataset',
                    help='training data or listfile path')
        self.parser.add_argument('--exp_dir',type=str, default='./exp',
                    help='root experimental directory')
        self.parser.add_argument('--exp_id', type=str, default='',
                    help='experimental name')
        self.parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu ids: e.g. 0  0,1,2, 0,2') ##TO DO
        self.parser.add_argument('--nThreads', type=int, default=4,
                    help='number of data loading threads')
        self.parser.add_argument('--is_train', type=bool, default=False,
                    help='training mode')
        self.parser.add_argument('--use_visdom', type=bool, default=True,
                    help='use visdom to display')
        self.parser.add_argument('--vis_env', type=str, default='main',
                    help='environment name for visdom')
        self.parser.add_argument('--use_html', type=bool, default=False,
                    help='use html to store images')
        self.parser.add_argument('--display_winsize', type=int, default=256,
                    help='display window size') ##TO DO
        self.parser.add_argument('--dataset', type=str, default='mpii',
                    help='dataset type')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # str_ids = self.opt.gpu_ids.split(',')
        # self.opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.exp_id == '':
            print('Please set the experimental ID with option --exp_id')
            exit()
        exp_dir = os.path.join(self.opt.exp_dir, self.opt.exp_id)
        util.mkdirs(exp_dir)
        if self.opt.resume_prefix_pose != '':
            trunc_index = self.opt.resume_prefix_pose.index('pth')
            self.opt.resume_prefix_pose = self.opt.resume_prefix_pose[0:trunc_index - 1]
            self.opt.resume_prefix_pose += '-'
            # opt_name = self.opt.resume_prefix_pose + 'opt.txt'
            # opt_name = os.path.join(exp_dir, opt_name)
        # else:
        #     opt_name = os.path.join(exp_dir, 'opt.txt')
        if self.opt.resume_prefix_asn != '':
            trunc_index = self.opt.resume_prefix_asn.index('pth')
            self.opt.resume_prefix_asn = self.opt.resume_prefix_asn[0:trunc_index - 1]
            self.opt.resume_prefix_asn += '-'
            # opt_name = self.opt.resume_prefix_asdn + 'opt.txt'
            # opt_name = os.path.join(exp_dir, opt_name)
        # else:
        #     opt_name = os.path.join(exp_dir, 'opt.txt')
        if self.opt.resume_prefix_dropout != '':
            trunc_index = self.opt.resume_prefix_dropout.index('pth')
            self.opt.resume_prefix_dropout = self.opt.resume_prefix_dropout[0:trunc_index - 1]
            self.opt.resume_prefix_dropout += '-'
        if self.opt.resume_prefix_aug != '':
            trunc_index = self.opt.resume_prefix_aug.index('pth')
            self.opt.resume_prefix_aug = self.opt.resume_prefix_aug[0:trunc_index - 1]
            self.opt.resume_prefix_aug += '-'
        with open('opt.txt', 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
