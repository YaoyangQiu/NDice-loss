import os
import numpy as np
import cv2


image_pth = r'E:\data\PSFH_Dataset\images\train'
cases = [i for i in os.listdir(image_pth) if i.endswith('.png')]
image = np.zeros((256, 256, 3200), dtype='uint8')
for i in range(len(cases)):
    image[:, :, i] = cv2.imread(os.path.join(image_pth, cases[i]), flags=cv2.IMREAD_GRAYSCALE)
data_mean = np.mean(image)
data_std = np.std(image)
print(data_mean, data_std)