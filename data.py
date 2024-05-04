import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from cellpose import dynamics
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from utils import coco_to_masklist


class CellFlowDataset(Dataset):
    def __init__(
        self, coco_annotation_file: str, image_dir: str, transform: transforms.Compose
    ):
        self.coco = COCO(coco_annotation_file)
        self.image_dir = image_dir

        print("Building the masks from the annotation file")
        mask_list = coco_to_masklist(self.coco)

        print("Building the flows from the masks")
        self.flow = dynamics.labels_to_flows(mask_list)
        self.transform = transform

    def __len__(self):
        return len(self.flow)

    def __getitem__(self, idx):
        image_name = self.coco.loadImgs(self.coco.getImgIds()[idx])[0]["file_name"]
        image = Image.open(os.path.join(self.image_dir, image_name))
        image = transforms.ToTensor()(image)
        flow = torch.tensor(self.flow[idx][1:, :, :])

        image = torch.cat((image, flow), dim=0)
        image = self.transform(image)
        image, flow = image[:3, :, :], image[3:, :, :]

        return image, flow


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.RandomVerticalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    dataset = CellFlowDataset(
        "/data/disk01/cell/cell_1360x1024/annotation/train.json",
        "/data/disk01/cell/cell_1360x1024/train",
        transform=transform,
    )
    image, flow = dataset[62]
    image = F.to_pil_image(image)
    plt.figure(figsize=(30, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(flow[0])
    plt.subplot(1, 3, 2)
    plt.imshow(flow[1])
    plt.subplot(1, 3, 3)
    plt.imshow(flow[2])
    plt.show()
