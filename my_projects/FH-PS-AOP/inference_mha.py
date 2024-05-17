import os
from mmseg.apis import init_model, inference_model
from medpy import metric
import cv2
import numpy as np
import torch
import SimpleITK


def multi_class_metrics(result, reference, target=None):
    classes = np.unique(result[result > 0]) if target is None else target
    output = []
    for i in classes:
        result_copy = np.zeros_like(result)
        result_copy[result == i] = 1
        reference_copy = np.zeros_like(reference)
        reference_copy[reference == i] = 1
        dice = metric.binary.dc(result_copy, reference_copy)
        asd = metric.binary.asd(result_copy, reference_copy)
        hd = metric.binary.hd(result_copy, reference_copy)
        output.append([dice, asd, hd])
    return output


config = 'upernet_r101_4xb4-aw-dice-3.0-ce-1.0-80k_psfh-256x256'
images_pth = r'E:\data\PSFH_Dataset\images\val'
labels_pth = r'E:\data\PSFH_Dataset\labels\val'
out_pth = r'E:\data\PSFH_Dataset\tta_inference_{}'.format(config)
if not os.path.exists(out_pth):
    os.makedirs(out_pth)

model_config = r'..\..\work_dirs\{}\{}.py'.format(config, config)
model_weight = r'..\..\work_dirs\{}\iter_80000.pth'.format(config)
model = init_model(model_config, model_weight)
# torch.save(model, 'test.pt')

results = dict(class_name=[], mDice=[], mASD=[], mHD=[], score=[])
images = [i for i in os.listdir(images_pth) if i.endswith('mha')]
total_metrics = {'class1':[], 'class2':[]}
for i in images:
    print(i)
    # pred = inference_model(model, os.path.join(images_pth, i))
    # img = cv2.imread(os.path.join(images_pth, i))
    img = SimpleITK.ReadImage(os.path.join(images_pth, i))
    img = SimpleITK.GetArrayFromImage(img)
    img = np.transpose(img, (1, 2, 0))
    pred1 = inference_model(model, img)
    # pred = pred.pred_sem_seg.numpy()
    pred1 = pred1.seg_logits.data
    pred1 = torch.softmax(pred1, dim=0)
    pred2 = inference_model(model, img[:, ::-1, :])
    pred2 = pred2.seg_logits.data
    pred2 = torch.softmax(pred2, dim=0)
    pred2 = torch.flip(pred2, dims=(2,))
    pred = pred1 + pred2
    pred = torch.argmax(pred, dim=0)

    pred = pred.cpu().numpy()
    # cv2.imwrite(os.path.join(out_pth, i), pred)
    # pred = cv2.imread(os.path.join(out_pth, i), cv2.IMREAD_GRAYSCALE)
    pred = pred.astype('uint8')
    pred = SimpleITK.GetImageFromArray(pred)
    SimpleITK.WriteImage(pred, os.path.join(out_pth, i))