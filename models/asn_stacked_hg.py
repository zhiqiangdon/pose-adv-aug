# Zhiqiang Tang, Feb 2017
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter


class _Residual(nn.Module):

    def __init__(self, in_num, out_num, adapter=None):
        super(_Residual, self).__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.conv1 = nn.Conv2d(in_num, out_num/2, kernel_size=1,
                               stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_num/2)
        self.conv2 = nn.Conv2d(out_num/2, out_num/2, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_num/2)
        self.conv3 = nn.Conv2d(out_num/2, out_num, kernel_size=1,
                               stride=1, bias=True)
        self.bn3 = nn.BatchNorm2d(out_num)
        self.relu = nn.ReLU(inplace=True)

        self.adapter = adapter

    def forward(self, x):
        if self.adapter is None:
            shortcut = x
        else:
            shortcut = self.adapter(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out += shortcut
        out = self.bn3(out)
        out = self.relu(out)

        return out

class _Hourglass(nn.Module):
    def __init__(self, chan, num_modules):
        super(_Hourglass, self).__init__()
        self.num_modules = num_modules
        self.chan = chan
        self.down1 = self._stack_residual()
        self.down2 = self._stack_residual()
        self.down3 = self._stack_residual()
        self.down4 = self._stack_residual()
        self.up1 = self._stack_residual()
        self.up2 = self._stack_residual()
        self.up3 = self._stack_residual()
        self.up4 = self._stack_residual()
        self.skip1 = self._stack_residual()
        self.skip2 = self._stack_residual()
        self.skip3 = self._stack_residual()
        self.skip4 = self._stack_residual()
        self.neck = self._stack_residual()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.keys = ['neck', 'skip1', 'skip2', 'skip3', 'skip4']

    def _stack_residual(self):
        residual_stack = []
        for i in range(0, self.num_modules):
            residual_stack.append(_Residual(self.chan, self.chan))
        return nn.Sequential(*residual_stack)

    def _dropout(self, x, masks):
        # x: n x c x h x w
        # masks: n x 1 x 4 x 4
        # sample_num = x.data.size(0)
        height = x.data.size(2)
        width = x.data.size(3)
        assert(height == width)
        scale = height / 4
        # print(scale)
        # assert (len(ys) == len(xs))
        # masks = masks.unsqueeze(1)
        # print(masks.size())
        # masks = 1 - masks
        # print(masks[0])
        if scale != 1:
            masks = F.upsample_nearest(masks, scale_factor=scale)
        # print(masks[0])
        # print(x[0, 1])
        x = x * masks.expand(x.size())
        # print(x[0, 1])
        # exit()
        return x

    def _sample_mask(self, pred_masks):
        dropout_num = 2
        assert len(pred_masks.size()) == 4
        sample_num = pred_masks.size(0)
        chan_num = pred_masks.size(1)
        assert chan_num == 1
        height = pred_masks.size(2)
        width = pred_masks.size(3)
        assert height == width
        # print pred_masks.size()
        dropout_masks = torch.ones(pred_masks.size()).cuda()
        all_size = height * width
        # print all_size
        probs = F.softmax(pred_masks.view(sample_num, -1)).data.cpu().numpy()
        # ys = torch.zeros(sample_num, dropout_num)
        # xs = torch.zeros(sample_num, dropout_num)
        indexes = torch.zeros(sample_num, dropout_num).long()
        for i in range(0, sample_num):
            # per_pred_mask = pred_masks[i]
            # prob = softmax(per_pred_mask.view(1, -1)).data.cpu().numpy()
            # print probs[i]
            dropout_indexes = np.random.choice(all_size, dropout_num, p=probs[i], replace=False)
            # print dropout_indexes
            for j in range(0, len(dropout_indexes)):
                y = dropout_indexes[j] / width
                x = dropout_indexes[j] % width
                # print y, x
                dropout_masks[i, 0, y, x] = 0
                # ys[i, j] = y
                # xs[i, j] = x
                indexes[i, j] = dropout_indexes[j]
                # print dropout_masks
        # exit()
        dropout_masks = torch.autograd.Variable(dropout_masks, requires_grad=False)
        return dropout_masks, indexes


    def forward(self, x, asn=None, is_half_hg=False, is_aug=False, is_dropout=False, dropout_masks=None):
        skip_x1 = self.skip1(x)

        x = self.maxpool(x)
        x = self.down1(x)
        skip_x2 = self.skip2(x)

        x = self.maxpool(x)
        x = self.down2(x)
        skip_x3 = self.skip3(x)

        x = self.maxpool(x)
        x = self.down3(x)
        skip_x4 = self.skip4(x)

        x = self.maxpool(x)
        x = self.down4(x)

        x = self.neck(x)

        if asn is not None:
            assert dropout_masks is None
            asn_input = {'neck': x.detach(), 'skip1': skip_x1.detach(),
                          'skip2': skip_x2.detach(), 'skip3': skip_x3.detach(), 'skip4': skip_x4.detach()}
            for key in self.keys:
                asn_input[key].volatile = False
            assert is_aug != is_dropout
            if is_aug:
                pred_scale_distri, pred_rotation_distri = asn(asn_input, is_aug=True)
                # pred_scale_distri, pred_rotation_distri = asn(asn_input)
                if is_half_hg:
                    # print 'hg augmentation half hg'
                    return pred_scale_distri, pred_rotation_distri
            else:
                pred_masks = asn(asn_input, is_dropout=True)
                if is_half_hg:
                    # print 'hg dropout half hg'
                    return pred_masks
                else:
                    dropout_masks, indexes = self._sample_mask(pred_masks)
                    x = self._dropout(x, dropout_masks)
                    skip_x1 = self._dropout(skip_x1, dropout_masks)
                    skip_x2 = self._dropout(skip_x2, dropout_masks)
                    skip_x3 = self._dropout(skip_x3, dropout_masks)
                    skip_x4 = self._dropout(skip_x4, dropout_masks)
        elif dropout_masks is not None:
            assert asn is None
            x = self._dropout(x, dropout_masks)
            skip_x1 = self._dropout(skip_x1, dropout_masks)
            skip_x2 = self._dropout(skip_x2, dropout_masks)
            skip_x3 = self._dropout(skip_x3, dropout_masks)
            skip_x4 = self._dropout(skip_x4, dropout_masks)

        x = self.up4(x)
        x = self.upsample(x)
        x = x + skip_x4
        x = self.up3(x)
        x = self.upsample(x)
        x = x + skip_x3
        x = self.up2(x)
        x = self.upsample(x)
        x = x + skip_x2
        x = self.up1(x)
        x = self.upsample(x)
        x = x + skip_x1

        if asn is not None:
            if is_aug:
                # print 'hg augmentation whole hg'
                return x, pred_scale_distri, pred_rotation_distri
            else:
                # print 'hg dropout whole hg'
                return x, pred_masks, indexes, dropout_masks
        else:
            return x

class _Hourglass_Wrapper(nn.Module):
    def __init__(self, num_modules, num_stacks, chan=256, num_classes=16):
        print('stack number is %d' % num_stacks)
        print('module number is %d' % num_modules)
        print('channel number is %d' % chan)
        super(_Hourglass_Wrapper, self).__init__()
        self.chan = chan
        self.num_modules = num_modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.residual1 = self._make_adapter_residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual2 = _Residual(128, 128)
        self.residual3 = self._make_adapter_residual(128, chan)

        self.hg = []
        self.post_res = []
        self.linear = []
        self.out_conv = []
        self.forth_conv = []
        self.in_conv = []
        self.num_stacks = num_stacks
        for i in range(0, num_stacks):
            self.hg.append(_Hourglass(chan=chan, num_modules=num_modules))
            self.post_res.append(self._stack_residual())
            self.linear.append(nn.Sequential(
                nn.Conv2d(chan, chan, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(chan),
                nn.ReLU(inplace=True)))
            self.out_conv.append(nn.Conv2d(chan, num_classes, kernel_size=1, stride=1, bias=True))
            if i < num_stacks-1:
                self.forth_conv.append(nn.Conv2d(chan, chan, kernel_size=1, stride=1, bias=True))
                self.in_conv.append(nn.Conv2d(num_classes, chan, kernel_size=1, stride=1, bias=True))

        self.hg = nn.ModuleList(self.hg)
        self.post_res = nn.ModuleList(self.post_res)
        self.linear = nn.ModuleList(self.linear)
        self.out_conv = nn.ModuleList(self.out_conv)
        self.forth_conv = nn.ModuleList(self.forth_conv)
        self.in_conv = nn.ModuleList(self.in_conv)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stdv = 1/math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
                    # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                # m.weight.data.zero_()
                m.bias.data.zero_()

    def _stack_residual(self):
        residual_stack = []
        for i in range(0, self.num_modules):
            residual_stack.append(_Residual(self.chan, self.chan))
        return nn.Sequential(*residual_stack)

    def _make_adapter_residual(self, in_num, out_num):
        adapter = nn.Conv2d(in_num, out_num, kernel_size=1, stride=1, bias=True)
        return _Residual(in_num, out_num, adapter)

    def forward(self, x, asn=None, is_half_hg=False, is_aug=False, is_dropout=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.maxpool(x)
        x = self.residual2(x)
        x = self.residual3(x)

        out = []
        for i in range(0, self.num_stacks):
            # print('hg %d' % i)
            if i == 0:
                if asn is not None:
                    print 'adversarial hg ', i
                    assert is_aug != is_dropout
                    if is_aug:
                        print 'augmentation'
                        if is_half_hg:
                            print 'half hg'
                            pred_scale_distri, pred_rotation_distri =\
                                self.hg[i](x, asn=asn, is_half_hg=is_half_hg, is_aug=is_aug)
                            return pred_scale_distri, pred_rotation_distri
                        else:
                            print 'whole hg'
                            y, pred_scale_distri, pred_rotation_distri = self.hg[i](x, asn=asn, is_aug=is_aug)
                    elif is_dropout:
                        print 'dropout'
                        if is_half_hg:
                            print 'half hg'
                            pred_mask = self.hg[i](x, asn=asn, is_half_hg=is_half_hg, is_dropout=is_dropout)
                            return pred_mask
                        else:
                            print 'whole hg'
                            y, pred_mask, indexes, dropout_masks = self.hg[i](x, asn=asn, is_dropout=is_dropout)
                else:
                    # print 'regular hg ', i
                    y = self.hg[i](x)
            else:
                if is_dropout:
                    print 'dropout hg ', i
                    y = self.hg[i](x, dropout_masks=dropout_masks)
                else:
                    # print 'regular hg ', i
                    y = self.hg[i](x)
            y = self.post_res[i](y)
            y = self.linear[i](y)
            tmp_out = self.out_conv[i](y)
            out.append(tmp_out)
            if i < self.num_stacks - 1:
                y = self.forth_conv[i](y)
                tmp_in = self.in_conv[i](tmp_out)
                x = x + y + tmp_in

        if asn is not None:
            if is_aug:
                return out, pred_scale_distri, pred_rotation_distri
            else:
                return out, pred_mask, indexes
        else:
            return out

def create_hg(num_stacks, num_modules, num_classes, chan):
    net = _Hourglass_Wrapper(num_stacks=num_stacks, num_modules=num_modules,
                             num_classes=num_classes, chan=chan)
    return net

class ASN(nn.Module):
    def __init__(self, chan_in, chan_out, scale_num, rotation_num, is_aug=False, is_dropout=False):
        assert is_aug != is_dropout
        if is_aug:
            print 'scale number is', scale_num
            print 'rotation number is', rotation_num
        super(ASN, self).__init__()
        self.num_modules = 3
        self.chan_in = chan_in
        self.chan_out = chan_out
        # self.is_astn = is_astn
        # self.is_asdn = is_asdn
        self.residual_skip1 = _Residual(chan_in, chan_out)
        self.residual_skip2 = _Residual(chan_in, chan_out)
        self.residual_skip3 = _Residual(chan_in, chan_out)
        self.residual_skip4 = _Residual(chan_in, chan_out)
        self.residual_neck = _Residual(chan_in, chan_out)
        self.merge1 = _Residual(chan_out, chan_out)
        self.merge2 = _Residual(chan_out, chan_out)
        self.merge3 = _Residual(chan_out, chan_out)
        self.merge4 = _Residual(chan_out, chan_out)
        self.deep_merge = self._stack_residual()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if is_aug:
            # self.out_conv = nn.Conv2d(chan_out, 1, kernel_size=1, stride=1, bias=True)
            self.avgpool = nn.AvgPool2d(4)
            self.fc_scale = nn.Linear(chan_out, scale_num)
            self.fc_rotation = nn.Linear(chan_out, rotation_num)
        if is_dropout:
            self.out_conv = nn.Conv2d(chan_out, 1, kernel_size=1, stride=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stdv = 1/math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
                    # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                # m.weight.data.zero_()
                m.bias.data.zero_()

    def _stack_residual(self):
        residual_stack = []
        for i in range(0, self.num_modules):
            residual_stack.append(_Residual(self.chan_out, self.chan_out))
        return nn.Sequential(*residual_stack)

    def forward(self, x, is_aug=False, is_dropout=False):
        skip1 = self.residual_skip1(x['skip1'])
        skip2 = self.residual_skip2(x['skip2'])
        skip3 = self.residual_skip3(x['skip3'])
        skip4 = self.residual_skip4(x['skip4'])
        neck = self.residual_neck(x['neck'])
        x = self.maxpool(skip1)
        x = x + skip2
        x = self.merge1(x)
        x = self.maxpool(x)
        x = x + skip3
        x = self.merge2(x)
        x = self.maxpool(x)
        x = x + skip4
        x = self.merge3(x)
        x = self.maxpool(x)
        x = x + neck
        x = self.merge4(x)
        # x = self.deep_merge1(x)
        # x = self.deep_merge2(x)
        x = self.deep_merge(x)
        # out = self.out_conv(x)
        # print x.size()
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # # print x.size()
        # scale_distri = self.fc_scale(x)
        # rotation_distri = self.fc_rotation(x)
        # return scale_distri, rotation_distri
        if is_aug:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # print x.size()
            scale_distri = self.fc_scale(x)
            rotation_distri = self.fc_rotation(x)
            return scale_distri, rotation_distri
        if is_dropout:
            mask_distri = self.out_conv(x)
            return mask_distri

def create_asn(chan_in, chan_out, scale_num=None, rotation_num=None, is_aug=False, is_dropout=False):
    net = ASN(chan_in=chan_in, chan_out=chan_out, scale_num=scale_num,
              rotation_num=rotation_num, is_aug=is_aug, is_dropout=is_dropout)
    return net


