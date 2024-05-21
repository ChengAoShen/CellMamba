import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
from cellpose import dynamics
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def create_mask_layer(coco: COCO, image_id: int) -> np.ndarray:
    """Create a mask layer from COCO annotations for a single image.

    Args:
        coco (COCO): COCO object
        image_id (int): Image ID

    Returns:
        np.ndarray: Mask layer
    """
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
    height, width = coco.imgs[image_id]["height"], coco.imgs[image_id]["width"]
    mask_layer = np.zeros((height, width), dtype=np.int32)

    instance_id = 1
    for ann in annotations:
        if "segmentation" in ann:
            # 为每个实例创建mask
            rle = mask_utils.frPyObjects(ann["segmentation"], height, width)
            mask = mask_utils.decode(rle)[:, :, 0]  # type:ignore
            # 将实例区域赋予不同的标签
            mask_layer[mask > 0] = instance_id
            instance_id += 1

    return mask_layer


def coco_to_masklist(coco: COCO) -> list:
    """Convert COCO annotations to mask list.

    Args:
        coco (COCO): COCO object

    Returns:
        list: Mask list
    """
    mask_list = []
    for image_id in tqdm(coco.getImgIds()):
        mask_list.append(create_mask_layer(coco, image_id))
    return mask_list


def show_mask(annotation_file: str, image_dir: str, img_id: int) -> None:
    coco = COCO(annotation_file)

    # 获取指定 ID 的图像信息
    img_info = coco.loadImgs([img_id])[0]
    img_filename = img_info["file_name"]
    image_path = os.path.join(image_dir, img_filename)

    # 加载图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 获取标注信息
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    # 将标注画在图像上
    plt.imshow(img)
    coco.showAnns(anns)
    plt.axis("off")  # 不显示坐标轴
    plt.show()


def visualize_and_save(image, masks, image_path):
    image = np.array(image.convert("RGB"))
    overlay = image.copy()
    for index in np.unique(masks):
        if index == 0:
            continue
        mask = masks == index
        colored_mask = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        overlay[mask] = overlay[mask] * 0.5 + colored_mask * 0.5

    combined = Image.fromarray(overlay.astype(np.uint8))
    combined.save(image_path)


if __name__ == "__main__":
    coco_annotation_file = "/data/disk01/cell/cell_1360x1024/annotation/train.json"
    coco = COCO(coco_annotation_file)
    mask_list = coco_to_masklist(coco)
    flow = dynamics.labels_to_flows(mask_list)
