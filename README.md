# Jointly Optimize Data Augmentation and Network Training: Adversarial Data Augmentation in Human Pose Estimation

Training and testing code for the paper 
**[Jointly Optimize Data Augmentation and Network Training: Adversarial Data Augmentation in Human Pose Estimation](https://arxiv.org/pdf/1805.09707.pdf)**, CVPR 2018

## Overview
Traditional random augmentation has two limitations. It doesn't consider the individual difference of training samples when doing augmentation. And it also independent of the training status of the target network. To tackle these problems, we design an agent to learn how to do data augmentation. We model the training process as an adversarial learning problem. The agent (generator), conditioning on the individual samples and network status, tries to generate ''hard'' augmentations for the target network. The target network, on the other hand, tries to learn better from the augmentations.


### Prerequisites

This package has the following requirements:

* `Python 2.7`
* `Pytorch 0.3.0.post4`


### Installing

Install pytorch:
```
pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
```
Install torchvision, scipy, matplotlib, dominate and visdom:
```
pip install torchvision scipy matplotlib dominate visdom
```

## Training

The training is divided into three stages. First, we pretrain the pose network for 10 epochs. Then we use the fixed pose network to pretrain the augmentation agent. Finally, we jointly optimize these two.

### 1. Pretrain the Pose Network

```
python stack-hg.py --gpu_id 0 --exp_id stack-2-hgs --vis_env stack-2-hgs --is_train true --bs 24
```

### 2. Pretrain the Augmentation Agent

Use the pose network to collect the scale and rotation distributions to train the agent:

```
python collect-scale-ditri.py --gpu_id 0 --exp_id stack-2-hgs --load_prefix_pose lr-0.00025-10.pth.tar --bs 10
```
```
python collect-rotation-ditri.py --gpu_id 0 --exp_id stack-2-hgs --load_prefix_pose lr-0.00025-10.pth.tar --bs 10
```
Pretrain the agent:
```
python pretrain-s-r-agent.py --gpu_id 0 --exp_id stack-2-hgs --load_prefix_pose lr-0.00025-10.pth.tar --bs 24
```
### 3. Jointly train the pose network and agent

```
python joint-train-pose-s-r-agent.py --gpu_id 0 --exp_id stack-2-hgs --load_prefix_pose lr-0.00025-10.pth.tar --load_prefix_sr lr-0.00025-1.pth.tar --vis_env stack-2-hgs-joint --is_train true --bs 24 
```

## Citation
If you find this code useful in your research, please consider citing:

```
@inproceedings{peng2018jointly,
  title={Jointly optimize data augmentation and network training: Adversarial data augmentation in human pose estimation},
  author={Peng, Xi and Tang, Zhiqiang and Yang, Fei and Feris, Rogerio S and Metaxas, Dimitris},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2226--2234},
  year={2018}
}
```
