import numpy as np
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm
from cellpose import dynamics


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


if __name__ == "__main__":
    coco_annotation_file = "/data/disk01/cell/cell_1360x1024/annotation/train.json"
    coco = COCO(coco_annotation_file)
    mask_list = coco_to_masklist(coco)
    flow = dynamics.labels_to_flows(mask_list)

    # mask, p = dynamics.compute_masks(flow[0][2:], flow[0][1], flow_threshold=0.0, min_size=1, interp=False)
