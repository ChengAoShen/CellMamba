import argparse
import os
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from cellpose import dynamics
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import UNet
from utils import visualize_and_save


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    coco_gt = COCO(args.annotation_path)
    image_dir = args.image_dir

    model = UNet(use_Mamba=True).to(device)
    model.load_state_dict(torch.load(args.model_path))

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    detection_results = []
    id = 0

    with torch.no_grad():
        start_time = time.time()
        for image_id in coco_gt.getImgIds():
            image_info = coco_gt.loadImgs(image_id)[0]
            image_name = image_info["file_name"]
            print(f"Predict image {image_name}...")

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
                min_size=0,
            )

            print(f"Mask num: {np.unique(masks).shape[0]}")
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
                    "bbox": list(
                        maskUtils.toBbox(maskUtils.encode(np.asfortranarray(mask)))
                    ),
                }
                detection_results.append(det)
                id += 1
            visualize_and_save(
                Image.open(os.path.join(image_dir, image_name)),
                masks,
                f"{args.output_dir}/{image_name}",
            )
        average_time = (time.time() - start_time) / len(coco_gt.getImgIds())
        print(f"Time for each image: {average_time:.2f}s")

    coco_dt = coco_gt.loadRes(detection_results)  # type:ignore
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.maxDets = [args.maxDets, args.maxDets, args.maxDets]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model on dataset."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="/data/disk01/MyoV/COCO/Train/Model_1_Early_Stage_Train_Data.json",
        help="Path to the COCO annotation JSON file.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/data/disk01/MyoV/COCO/Train/Image",
        help="Directory containing the images to process.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Size to which the images will be resized.",
    )
    parser.add_argument(
        "--maxDets",
        type=int,
        default=2000,
        help="Maximum detections for COCO evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/MyoV",
        help="Directory to save the output images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args)
