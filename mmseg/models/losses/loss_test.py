from mmseg.models.losses import CrossEntropyLoss, SurfaceLoss, DiceLoss, AutoWeightedCELoss
from mmseg.models.losses.auto_weighted_ce_loss import one_hot2dist, one_hot2weight, \
    one_hot2class_weight, one_hot2weight_s
import torch
import cv2
import torch.nn.functional as F
import numpy as np


if __name__ == '__main__':
    # loss_func = AutoWeightedCELoss()
    # input = torch.randn(3, 5, 20, 20, requires_grad=False)
    # input[0, 1, ...] = 1000.
    # target = torch.randint(5, (3, 20, 20), dtype=torch.int64)
    # target[0] = torch.argmax(input[0], dim=0)
    # input = torch.randn()
    # label = torch.tensor([[[0]]])
    # loss = loss_func(input, target)
    img = cv2.imread(r'E:\data\PSFH_Dataset\labels\train\00001.png', cv2.IMREAD_GRAYSCALE)
    # img = img[32:128+32, 64:128+64]
    # img = np.zeros((128, 128), dtype='uint8')
    # img[32:96, 32:96] = 1
    img_view = (img-img.min())/(img.max()-img.min()) * 255
    img = torch.from_numpy(img[None, :, :])
    one_hot_target = F.one_hot(
        torch.clamp(img.long(), 0, 3 - 1),
        num_classes=3)
    one_hot_target = torch.permute(one_hot_target, [0, 3, 1, 2])
    one_hot_target = one_hot_target.cuda()
    neighbor = one_hot2weight(one_hot_target)
    neighbor = neighbor.cpu().numpy()[0]
    neighbor = (neighbor - neighbor.min()) / (neighbor.max() - neighbor.min()) * 255
    dist = one_hot2dist(one_hot_target)
    dist = dist.cpu().numpy()[0]
    dist = (dist - dist.min()) / (dist.max() - dist.min()) * 255
    neighbor_s = one_hot2weight_s(one_hot_target)
    neighbor_s = neighbor_s.cpu().numpy()[0]
    neighbor_s = (neighbor_s-neighbor_s.min())/(neighbor_s.max()-neighbor_s.min()) * 255
    out = np.concatenate((img_view, neighbor, neighbor_s, dist), axis=1)
    out = out.astype('uint8')
    out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    cv2.imwrite('test.png', out)
    # pass
