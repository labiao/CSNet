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
    for file_name in tqdm(os.listdir(args.mask)):
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

            mask_ = np.zeros((256, 256))
            for mask1 in masks:
                mask1 = mask1.cpu().numpy()
                # mask1 = np.array(mask1)
                mask1 = mask1.astype(np.uint8)[0, :, :]
                mask1[mask1 == 1] = 255
                mask_ = mask_ + mask1

        # overlaps
        count = np.sum(mask == mask_)
        total_pixels = mask_.shape[0] * mask_.shape[1]
        overlap = count / total_pixels
        if overlap > 0.9:
            pass
        else:
            mask_ = mask

        # 保存影像
        save_file_name = f'{file_name}'
        save_path1 = os.path.join(save_path, save_file_name)
        cv2.imwrite(save_path1, mask_)


if __name__ == '__main__':
    SAM(None)
