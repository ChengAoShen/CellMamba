import copy
import json
import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from cellpose import dynamics
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
def visualize_and_save(image, masks, image_path):
    """
    可视化并保存图像和其对应的分割掩模。
    
    Args:
    image (PIL.Image): 原始图像。
    masks (np.ndarray): 分割掩模数组。
    image_path (str): 输出图像的保存路径。
    """
    image = np.array(image.convert('RGB'))  # 确保图像为RGB
    overlay = image.copy()
    for index in np.unique(masks):
        if index == 0:
            continue
        mask = (masks == index)
        colored_mask = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        overlay[mask] = overlay[mask] * 0.5 + colored_mask * 0.5

    combined = Image.fromarray(overlay.astype(np.uint8))
    combined.save(image_path)


model_path = "./checkpoint/best.pth"
coco_gt = COCO("/data/disk01/cell/cell_1360x1024/annotation/val_without_TBM1.json")
# coco_gt = COCO("/data/disk01/cell/cell_1360x1024/annotation/val.json")
image_dir = "/data/disk01/cell/cell_1360x1024/val"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path).to(device)

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ]
)

detection_results = []
id = 0

start_time = time.time()
for image_id in coco_gt.getImgIds():
    image_info = coco_gt.loadImgs(image_id)[0]
    image_name = image_info["file_name"]
    print(f"predict image {image_name}...")

    image = Image.open(os.path.join(image_dir, image_name))
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    output = torch.nn.functional.interpolate(
        output,
        size=(image_info["height"], image_info["width"]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    cell_prob = torch.sigmoid(output[0]).detach().cpu().numpy()
    flow = torch.tanh(output[1:3]).detach().cpu().numpy()
    masks, p = dynamics.compute_masks(
        flow,
        cell_prob,
        flow_threshold=0.0,
        min_size=100,
    )

    print(f"mask num:{np.unique(masks).shape[0]}")
    for index in np.unique(masks):
        if index == 0:
            continue
        mask = (masks == index).astype(np.uint8)
        score = cell_prob[mask == 1].mean()
        det = {
            "id": id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": maskUtils.encode(np.asfortranarray(mask)),
            "score": score,
            "bbox": list(maskUtils.toBbox(maskUtils.encode(np.asfortranarray(mask)))),
        }
        detection_results.append(det)
        id += 1
    visualize_and_save(Image.open(os.path.join(image_dir, image_name)), masks, f"./output/{image_name}")
average_time= (time.time()-start_time)/len(coco_gt.getImgIds())
print(f"Time for each image: {average_time:.2f}s")

coco_dt = coco_gt.loadRes(detection_results)
coco_eval = COCOeval(coco_gt, coco_dt, "segm")
coco_eval.params.maxDets = [2000, 2000, 2000]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()