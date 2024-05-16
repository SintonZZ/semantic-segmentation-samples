import cv2
import numpy as np
import torch
from torch.nn.parallel import DataParallel
from glob import glob
from arch.dlinknet import DinkNet34


device_ids = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0')
val_img_paths = glob('./dataset/valid/*sat.jpg')

model = DinkNet34().to(device)
model = DataParallel(model, device_ids=device_ids[:4])
model.load_state_dict(torch.load('./checkpoint/dlinknet34_none/dlinknet34_best.pth'))
model.eval()

ori_img = cv2.imread(val_img_paths[10])

# single image test
img = ori_img.astype(np.float32) / 255.0 * 3.2 - 1.6
img = torch.Tensor(img.transpose(2, 0, 1)[None]).to(device)
pred_mask = model(img)
pred_mask = pred_mask.squeeze().cpu().data.numpy()
pred_mask[pred_mask>0.5] = 255
pred_mask[pred_mask<=0.5] = 0
pred_mask = pred_mask.astype(np.uint8)

# TTA
img90 = np.array(np.rot90(ori_img))
img2 = np.concatenate([ori_img[None],img90[None]])
img2_vflip = np.array(img2)[:,::-1]
img4 = np.concatenate([img2, img2_vflip])
img4_hflip = np.array(img4)[:,:,::-1]
img8 = np.concatenate([img4, img4_hflip]).transpose(0,3,1,2)
img8 = np.array(img8, np.float32)/255.0 * 3.2 -1.6
img8 = torch.Tensor(img8).to(device)

mask = model(img8).squeeze().cpu().data.numpy()
mask1 = mask[:4] + mask[4:,:,::-1]
mask2 = mask1[:2] + mask1[2:,::-1]
mask_tta = (mask2[0] + np.rot90(mask2[1])[::-1,::-1]) / 8
mask_tta[mask_tta>0.5] = 255
mask_tta[mask_tta<=0.5] = 0
mask_tta = np.concatenate([mask_tta[:,:,None],mask_tta[:,:,None],mask_tta[:,:,None]],axis=2)
mask_tta = mask_tta.astype(np.uint8)

cv2.namedWindow('img', 0)
cv2.imshow('img', ori_img)
cv2.namedWindow('mask', 0)
cv2.imshow('mask', pred_mask)
cv2.namedWindow('mask_tta', 0)
cv2.imshow('mask_tta', mask_tta)
cv2.waitKey(0)
    
