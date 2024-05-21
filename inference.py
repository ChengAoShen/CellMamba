import argparse
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

from model import UNet
from utils import visualize_and_save


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        # Convert to a flat list of points in COCO polygon format
        poly = contour.flatten().tolist()
        if len(poly) > 4:  # COCO requires at least three points for a polygon
            polygons.append(poly)
    return polygons


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = UNet(use_Mamba=True).to(device)
    model.load_state_dict(torch.load(args.model_path))

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    detected_results = []
    image_info = []
    id = 0
    with torch.no_grad():
        for image_id, image_name in enumerate(os.listdir(args.image_dir)):
            print(f"Current image: {image_name}")
            image_path = os.path.join(args.image_dir, image_name)
            image = Image.open(image_path)
            w, h = image.size
            image_tensor = transform(image).unsqueeze(0).to(device)

            start_time = time.time()
            output = model(image_tensor)
            output = torch.nn.functional.interpolate(
                output, size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
            cell_prob = torch.sigmoid(output[0]).detach().cpu().numpy()
            flow = torch.tanh(output[1:3]).detach().cpu().numpy()
            masks, p = dynamics.compute_masks(
                flow, cell_prob, flow_threshold=0.0, min_size=100
            )

            print(f"Time cost: {time.time() - start_time:.2f}s")
            print(f"Mask num: {len(np.unique(masks))}")
            if args.visualize:
                visualize_and_save(image, masks, f"output/global/{image_name}")

            if args.save_json:
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process images to detect cells using a U-Net model and save results."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing the images to process.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Size to which the images will be resized.",
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=True,
        help="Enable visualization of the results.",
    )
    parser.add_argument(
        "--save_json", type=bool, default=False, help="Save the results in a JSON file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    inference(args)
