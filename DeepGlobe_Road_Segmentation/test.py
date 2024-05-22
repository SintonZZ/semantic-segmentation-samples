import cv2
import numpy as np
import torch
from torch.nn.parallel import DataParallel
import torch.utils.data as data
from glob import glob
from tqdm import tqdm
from arch.dlinknet import DinkNet34
from data_loader import DeepGlobeRoadExtract
from metrics import intersection_over_union, mIOU


device_ids = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0')


model = DinkNet34().to(device)
model = DataParallel(model, device_ids=device_ids[:2])
model.load_state_dict(torch.load('./checkpoint/dlinknet34_none/dlinknet34_best.pth'))
model.eval()

iou_score = 0.0
test_set = DeepGlobeRoadExtract(mode='test')
test_loader = data.DataLoader(test_set, batch_size=2, num_workers=2, pin_memory=True)
for data, mask in tqdm(test_loader):
    data = data.to(device)
    mask = mask.squeeze().numpy()
    pred_mask = model(data)
    pred_mask = pred_mask.squeeze().cpu().data.numpy()
    pred_mask[pred_mask>0.5] = 1
    pred_mask[pred_mask<=0.5] = 0

    iou = intersection_over_union(mask.reshape(-1), pred_mask.reshape(-1))[1]
    iou_score += iou
iou_score /= len(test_loader)
print(f"IOU: {iou_score:.3f}")
	
# # single image test
# val_img_paths = glob('./dataset/valid/*sat.jpg')
# ori_img = cv2.imread(val_img_paths[10])
# img = ori_img.astype(np.float32) / 255.0 * 3.2 - 1.6
# img = torch.Tensor(img.transpose(2, 0, 1)[None]).to(device)
# pred_mask = model(img)
# pred_mask = pred_mask.squeeze().cpu().data.numpy()
# pred_mask[pred_mask>0.5] = 255
# pred_mask[pred_mask<=0.5] = 0
# pred_mask = pred_mask.astype(np.uint8)

# # TTA
# img90 = np.array(np.rot90(ori_img))
# img2 = np.concatenate([ori_img[None],img90[None]])
# img2_vflip = np.array(img2)[:,::-1]
# img4 = np.concatenate([img2, img2_vflip])
# img4_hflip = np.array(img4)[:,:,::-1]
# img8 = np.concatenate([img4, img4_hflip]).transpose(0,3,1,2)
# img8 = np.array(img8, np.float32)/255.0 * 3.2 -1.6
# img8 = torch.Tensor(img8).to(device)

# mask = model(img8).squeeze().cpu().data.numpy()
# mask1 = mask[:4] + mask[4:,:,::-1]
# mask2 = mask1[:2] + mask1[2:,::-1]
# mask_tta = (mask2[0] + np.rot90(mask2[1])[::-1,::-1]) / 8
# mask_tta[mask_tta>0.5] = 255
# mask_tta[mask_tta<=0.5] = 0
# mask_tta = np.concatenate([mask_tta[:,:,None],mask_tta[:,:,None],mask_tta[:,:,None]],axis=2)
# mask_tta = mask_tta.astype(np.uint8)

# cv2.namedWindow('img', 0)
# cv2.imshow('img', ori_img)
# cv2.namedWindow('mask', 0)
# cv2.imshow('mask', pred_mask)
# cv2.namedWindow('mask_tta', 0)
# cv2.imshow('mask_tta', mask_tta)
# cv2.waitKey(0)
    
