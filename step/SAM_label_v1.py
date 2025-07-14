import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
import numpy as np
from skimage import io
from PIL import Image
import torch
import os

from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


def logit(x):
    a = torch.tensor(x)
    return torch.special.logit(a, eps=1e-6).numpy()


def inv_logit(x):
    return 0.5 * (1. + np.sign(x) * (2. / (1. + np.exp(-np.abs(x))) - 1.))


def get_target_bboxes(gray_img):
    coords = np.column_stack(np.where(gray_img == 255))
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h


def get_all_target_bboxes(gray_img):
    ret, thresh = cv2.threshold(gray_img, 2, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        target_boxes.append((x, y, x + w, y + h))
    return target_boxes


def SAM(args):
    # folderA_path = args.SAM_A
    # folderB_path = args.SAM_B
    folderA_path = './dataset/BCD/A1'
    folderB_path = './dataset/BCD/B1'
    mask_path = args.mask
    save_path = args.SAMlabel
    os.makedirs(save_path, exist_ok=True)

    sam_checkpoint = args.SAM_weight
    device = "cuda"
    model_type = "vit_h"
    sys.path.append("..")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    samPredictor = SamPredictor(sam)

    # for file_name in os.listdir(folderA_path):
    for file_name in (os.listdir(args.mask)):
        if file_name.endswith('.png'):
            mask = cv2.imread(os.path.join(mask_path, file_name), 0)
            imageA = cv2.imread(os.path.join(folderA_path, file_name))
            imageB = cv2.imread(os.path.join(folderB_path, file_name))
            mask = cv2.resize(mask, (256, 256))

            input_box = get_all_target_bboxes(mask)
            input_box = np.array(input_box)
            if len(input_box) == 0:
                continue

            image = imageA - imageB + 127
            samPredictor.set_image(image)

            input_box = torch.from_numpy(input_box)
            all_input_boxes = samPredictor.transform.apply_boxes_torch(input_box, imageA.shape[:2])

            masks, scores, logits = samPredictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=all_input_boxes.to(device=device),
                multimask_output=False,
            )
            # print(masks.shape, all_input_boxes.shape)  # 有多少个bounding box 就有多少个mask生成，融合是为了得到完整的mask
            mask_ = np.zeros((256, 256)).astype(np.uint8)
            for mask1 in masks:
                mask1 = mask1.cpu().numpy()
                # mask1 = np.array(mask1)
                mask1 = mask1.astype(np.uint8)[0, :, :]
                mask1[mask1 == 1] = 255
                mask_ = mask_ + mask1

        M_hat = (mask//255 + mask_//255) / 2  # [0.0, 0.5, 1.0]
        epsilon = 1e-10
        E_i = -M_hat * np.log(M_hat + epsilon) - (1 - M_hat) * np.log(1 - M_hat + epsilon)
        # print(np.unique(E_i))  # [-1.00000008e-10  6.93147180e-01]
        high_uncertainty_pixels = E_i > 0.6  # 没交集的像素
        U_a = np.mean(high_uncertainty_pixels.astype(int))
        # 不稳定像素占总像素比例低于 0.1
        tau_a = 0.1

        low_uncertainty_pixels = E_i < 0.6  # 有交集的像素（背景和前景）
        fg_pixels = M_hat > 0.51
        sig_IoU = np.sum(low_uncertainty_pixels & fg_pixels) / np.sum(M_hat > 0)  # 能够保留高IoU，和一些小的变化
        if U_a < tau_a and sig_IoU > 0.5:
            mask_ = M_hat * (1-E_i) * 255

            # 保存影像
            save_file_name = f'{file_name}'
            save_path1 = os.path.join(save_path, save_file_name)
            cv2.imwrite(save_path1, mask_.astype(int))
