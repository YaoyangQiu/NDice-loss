import os
import shutil
import random

import numpy as np

org_img_pth = r'E:\data\Pubic Symphysis-Fetal Head Segmentation and Angle of Progression\image_png'
org_label_pth = r'E:\data\Pubic Symphysis-Fetal Head Segmentation and Angle of Progression\label_png'
root_pth = r'E:\data\PSFH_Dataset'
image_pth = os.path.join(root_pth, 'images')
label_pth = os.path.join(root_pth, 'labels')
cases = [i for i in os.listdir(org_img_pth) if i.endswith('.png')]
random.seed(530)
random.shuffle(cases)
train_set = cases[:3200]
train_set.sort()
val_set = cases[3200:]
val_set.sort()

image_pth_tr = os.path.join(image_pth, 'train')
image_pth_ts = os.path.join(image_pth, 'val')
label_pth_tr = os.path.join(label_pth, 'train')
label_pth_ts = os.path.join(label_pth, 'val')
os.makedirs(image_pth_tr)
os.makedirs(image_pth_ts)
os.makedirs(label_pth_tr)
os.makedirs(label_pth_ts)

for i in train_set:
    shutil.copy(os.path.join(org_img_pth, i), os.path.join(image_pth_tr, i))
    shutil.copy(os.path.join(org_label_pth, i), os.path.join(label_pth_tr, i))

for i in val_set:
    shutil.copy(os.path.join(org_img_pth, i), os.path.join(image_pth_ts, i))
    shutil.copy(os.path.join(org_label_pth, i), os.path.join(label_pth_ts, i))