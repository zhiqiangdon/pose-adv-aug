# Zhiqiang Tang, May 2017
import os
import numpy as np
from PIL import Image, ImageDraw
import torch

############ read and write ##########
def ReadAnnotMPII(path):
    annot = {}
    with open(path, 'r') as fd:
        annot['imgName'] = next(fd).rstrip('\n')
        annot['headSize'] = float(next(fd).rstrip('\n'))
        annot['center'] = [int(x) for x in next(fd).split()]
        annot['scale'] = float(next(fd).rstrip('\n'))
        annot['pts'] = []
        annot['vis'] = []
        for line in fd:
            x, y, isVis = [int(float(x)) for x in line.split()]
            annot['pts'].append((x,y))
            annot['vis'].append(isVis)
    return annot

def DrawImgPts(img,pts):
    NLMK = pts.shape[0]
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for l in range(NLMK):
        if pts[l, 0] == 0 and pts[l, 1] == 0:
            continue
        draw.ellipse((pts[l,0]-3,pts[l,1]-3,pts[l,0]+3,pts[l,1]+3), fill='red')
    del draw
    return img_draw


############ pts and heatmap ##########
def pts2heatmap(pts, heatmap_shape, sigma=1):
    # generate heatmap n x res[0] x res[1], each row is one pt (x, y)
    heatmap = np.zeros((pts.shape[0], heatmap_shape[0], heatmap_shape[1]))
    valid_pts = np.zeros((pts.shape))
    for i in range(0, pts.shape[0]):
        if pts[i][0] <= 0 or pts[i][1] <= 0 or \
                        pts[i][0] > heatmap_shape[1] or pts[i][1] > heatmap_shape[0]:
            continue
        heatmap[i] = draw_gaussian(heatmap[i], pts[i], sigma)
        valid_pts[i] = pts[i]
    return heatmap, valid_pts

def pts2spatialmask(pts, heatmap_shape, mask_shape, pts_weights):
    assert pts.shape[0] == len(pts_weights)
    assert np.sum(pts_weights) == 1
    spatial_mask = np.zeros((mask_shape[0], mask_shape[1]))
    # print spatial_mask
    block_size_y = heatmap_shape[0] / mask_shape[0]
    block_size_x = heatmap_shape[1] / mask_shape[1]
    # print block_size_x, block_size_y
    # print pts
    for i in range(0, pts.shape[0]):
        # assert pts[i, 0] < heatmap_shape[0] and pts[i, 1] < heatmap_shape[1]
        if pts[i, 0] > 0 and pts[i, 0] < heatmap_shape[0] and pts[i, 1] > 0 and pts[i, 1] < heatmap_shape[1]:
            # print pts[i]
            # notice that pts[i, 0] is the x coordinate and pts[i, 1] is the y coordinate
            y = pts[i, 1] / block_size_y
            x = pts[i, 0] / block_size_x
            spatial_mask[y, x] += pts_weights[i]
            # print y,x
            # print spatial_mask
    # print 'before mask is', spatial_mask
    # before_mask = spatial_mask
    spatial_mask = spatial_mask / np.sum(spatial_mask)
    # print 'after mask is', spatial_mask
    # exit()
    # print 'after sum is ', np.sum(spatial_mask)
    # if np.sum(spatial_mask) != 1.:
    #     print np.sum(spatial_mask)
    #     print 'before mask is', before_mask
    #     print 'after mask is', spatial_mask
    #     exit()
    # assert np.sum(spatial_mask) == 1.
    # exit()
    return spatial_mask

def draw_gaussian(img, pt, sigma):
    # Draw a 2D gaussian
    tmp_size = np.ceil(3 * sigma)
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size), int(pt[1] + tmp_size)]
    # Check that any part of the gaussian is in-bounds
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (tmp_size ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0]+1, img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])
    g_y = max(0, -ul[1]), min(br[1]+1, img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])
    # Image range
    img_x = max(0, ul[0]), min(br[0]+1, img.shape[1])
    img_y = max(0, ul[1]), min(br[1]+1, img.shape[0])

    # # Usable gaussian range
    # g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    # g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # # Image range
    # img_x = max(0, ul[0]), min(br[0], img.shape[1])
    # img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def heatmap2pts(heatmap):
    # heatmap: b x n x h x w tensor
    # preds: b x n x 2 tensor
    max, idx = torch.max(heatmap.view(heatmap.size(0), heatmap.size(1), heatmap.size(2) * heatmap.size(3)), 2)
    # print('hahahah')
    # print(max)
    # print(idx)
    pts = torch.zeros(idx.size(0), idx.size(1), 2)
    pts[:, :, 0] = idx % heatmap.size(3)
    # preds[:, :, 0].add_(-1).fmod_(heatmap.size(3)).add_(1)
    # idx is longTensor type, so no floor function is needed
    pts[:, :, 1] = idx / heatmap.size(3)
    pts[:, :, 1].floor_().add_(0.5)
    # preds[:, :, 1].div_(heatmap.size(3)).floor_()
    predMask = max.gt(0).repeat(1, 1, 2).float()
    # print(preds.size())
    # print(predMask.size())
    pts = pts * predMask
    # print(preds[:, :, 0])
    return pts

def pts2resmap(pts, resmap_shape, radius):
    # generate multi-channel resmap, one map for each point
    pts_num = pts.shape[0]
    resmap = np.zeros((pts_num, resmap_shape[0], resmap_shape[1]))
    valid_pts = np.zeros((pts.shape))
    for i in range(0, pts_num):
        # if vis_arr[i] == -1:
        #     continue
        # note that here we can't use vis_arr to indicate whether to draw the annotation
        # because some pts are labeled visible but not within the effective crop area due to the
        # inaccurate person scale in the original annotation
        if pts[i][0] <= 0 or pts[i][1] <= 0 or \
                        pts[i][0] > resmap_shape[1] or pts[i][1] > resmap_shape[0]:
            continue
        y, x = np.ogrid[-pts[i][1]:resmap_shape[0] - pts[i][1], -pts[i][0]:resmap_shape[1] - pts[i][0]]
        mask = x * x + y * y <= radius * radius
        resmap[i][mask] = 1
        valid_pts[i] = pts[i]
        # print('channel %d sum is %.f' % (i, np.sum(resmap[i])))
    return resmap, valid_pts

def weights_from_grnd_maps(maps, fgrnd_weight, bgrnd_weight):
    # maps: c x h x w tensor, zero is background, maps could be resmap or heatmap
    # weights: c x h x w tensor
    weights = torch.ones(maps.size())
    per_map_sum = maps.size(1) * maps.size(2)
    factor = float(fgrnd_weight) / float(bgrnd_weight)
    for i in range(0, maps.size(0)):
        mask_foregrnd = maps[i] > 0
        foregrnd_pixel_num = mask_foregrnd.sum()
        if foregrnd_pixel_num == 0:
            continue
        per_weight = float(per_map_sum - foregrnd_pixel_num) / float(foregrnd_pixel_num) * factor
        weights[i][mask_foregrnd] = int(per_weight)

    return weights

if __name__=='__main__':
    print 'Python pts to landmark by Xi Peng'

