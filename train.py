import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from cellpose import dynamics
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import CellFlowDataset
from model import UNet


def visualize_predictions(epoch, output, image, flow, writer):
    mask_list = []
    num = min(4, image.size(0))
    for index in range(num):
        mask, p = dynamics.compute_masks(
            torch.tanh(output[index, 1:3, :, :]).detach().cpu().numpy(),
            torch.sigmoid(output[index, 0, :, :]).detach().cpu().numpy(),
            flow_threshold=0.0,
            min_size=0,
        )
        mask_list.append(mask)
    mask_list = np.array(mask_list, dtype=np.float32)
    mask = torch.tensor(mask_list, dtype=torch.float32).unsqueeze(1)

    writer.add_images("predict_image/mask", mask, epoch)
    writer.add_images("image/input", image[:num, :, :, :], epoch)
    writer.add_images(
        "predict_image/probability",
        torch.where(torch.sigmoid(output[:num, 0:1, :, :]) > 0.5, 1.0, 0),
        epoch,
    )
    writer.add_images("gt_image/probability", flow[:num, 0:1, :, :], epoch)
    writer.add_images(
        "predict_image/y_flow", (torch.tanh(output[:num, 1:2, :, :]) + 1) / 2, epoch
    )
    writer.add_images(
        "predict_image/x_flow", (torch.tanh(output[:num, 2:3, :, :]) + 1) / 2, epoch
    )
    writer.add_images("gt_image/y_flow", (flow[:num, 1:2, :, :] + 1) / 2, epoch)
    writer.add_images("gt_image/x_flow", (flow[:num, 2:3, :, :] + 1) / 2, epoch)


def train(args):
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Prepare the Dataset
    train_transform = transforms.Compose(
        [
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((args.img_size, args.img_size)),
        ]
    )

    train_dataset = CellFlowDataset(
        args.json_path, args.image_dir, transform=train_transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    length_train_dataset = len(train_dataset)

    # Load or initialize the model
    if args.parameters_path:
        model = UNet(use_Mamba=True).to(device)
        model.load_state_dict(torch.load(args.parameters_path))
    else:
        model = UNet(use_Mamba=True).to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.05)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss, mask_loss, flow_loss = 0.0, 0.0, 0.0

        for i, (image, flow) in enumerate(train_dataloader):
            image = image.to(device)
            flow = flow.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss1 = nn.BCEWithLogitsLoss()(output[:, 0:1, :, :], flow[:, 0:1, :, :])
            loss2 = torch.mean(
                torch.abs(torch.tanh(output[:, 1:3, :, :]) - flow[:, 1:3, :, :])
            )
            loss = 0.2 * loss1 + 0.8 * loss2
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            mask_loss += loss1.item()
            flow_loss += loss2.item()

        scheduler.step()

        # Log loss and learning rate
        average_loss = running_loss / length_train_dataset
        mask_loss = mask_loss / length_train_dataset
        flow_loss = flow_loss / length_train_dataset
        print(
            f"Epoch {epoch+1}, loss: {average_loss}, mask_loss: {mask_loss}, flow_loss: {flow_loss}"
        )
        writer.add_scalar("learning rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("loss/all average loss", average_loss, epoch)
        writer.add_scalar("loss/mask loss", mask_loss, epoch)
        writer.add_scalar("loss/flow loss", flow_loss, epoch)
        running_loss, mask_loss, flow_loss = 0.0, 0.0, 0.0

        if (epoch + 1) % args.visualize_interval == 0:
            visualize_predictions(epoch, output, image, flow, writer)

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), f"{args.checkpoint_dir}/model_{epoch+1}.pth")

    writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the model for cell segmentation and flow estimation."
    )
    parser.add_argument(
        "--epochs", type=int, default=240, help="Number of epochs to train."
    )
    parser.add_argument(
        "--img_size", type=int, default=512, help="Size of the input images."
    )
    parser.add_argument(
        "--visualize_interval", type=int, default=24, help="Interval for visualization."
    )
    parser.add_argument(
        "--save_interval", type=int, default=24, help="Interval for saving the model."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/data/disk01/MyoV/COCO/Train/Model_1_Early_Stage_Train_Data.json",
        help="Path to the JSON file containing annotations.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/data/disk01/MyoV/COCO/Train/Image",
        help="Directory containing the images.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--parameters_path", type=str, help="Path to the model parameters to load."
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD optimizer."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for SGD optimizer.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoint",
        help="Directory to save the model.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
