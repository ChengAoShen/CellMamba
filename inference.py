import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import UNet  # 确保这个模型已经正确定义
from cellpose import dynamics
import cv2
import json
from pycocotools import mask as maskUtils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./checkpoint/U_Mamba/best_85.pth"
image_dir = "/data/disk01/cell/cell_1360x1024/val"
model = UNet(use_Mamba=True).to(device)
model.load_state_dict(torch.load(model_path))

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ]
)


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        # 转换为COCO格式的扁平多边形列表
        poly = contour.flatten().tolist()
        if len(poly) > 4:  # COCO要求多边形至少有三个点
            polygons.append(poly)
    return polygons


# 处理图像
detected_results = []
image_info = []
id = 0
with torch.no_grad():
    for image_id, image_name in enumerate(os.listdir(image_dir)):
        print(f"current image: {image_name}")
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        w, h = image.size

        image_tensor = transform(image).unsqueeze(0).to(device)
        output = model(image_tensor)
        output = torch.nn.functional.interpolate(
            output, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
        cell_prob = torch.sigmoid(output[0]).detach().cpu().numpy()
        flow = torch.tanh(output[1:3]).detach().cpu().numpy()
        masks, p = dynamics.compute_masks(
            flow, cell_prob, flow_threshold=0.0, min_size=100
        )

        image_info.append(
            {"id": image_id, "width": w, "height": h, "file_name": image_name}
        )

        for index in np.unique(masks):
            if index == 0:
                continue
            mask = (masks == index).astype(np.uint8)
            det = {
                "id": id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": mask_to_polygons(mask),
                "bbox": list(
                    maskUtils.toBbox(maskUtils.encode(np.asfortranarray(mask)))
                ),
            }
            detected_results.append(det)
            id += 1

# 输出为JSON
categories = [{"id": 1, "name": "cell", "supercategory": "none"}]
with open("results.json", "w") as f:
    json.dump(
        {
            "images": image_info,
            "annotations": detected_results,
            "categories": categories,
        },
        f,
        indent=4,
    )
