import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from cellpose import dynamics
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data import CellFlowDataset
from model import UNet

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Perpare the Dataset
train_transform = transforms.Compose(
    [
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((1024, 1024)),
    ]
)

train_dataset = CellFlowDataset(
    "/data/disk01/cell/cell_1360x1024/annotation/train.json",
    "/data/disk01/cell/cell_1360x1024/train",
    transform=train_transform,
)


train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
length_train_dataset = len(train_dataset)


model = UNet(use_Mamba=True).to(device)
model.load_state_dict(torch.load("./checkpoint/U_Mamba/best_85.pth"))

# model = torch.load("./checkpoint/best.pth").to(device)

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001
)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.05)


# Training
epochs = 100
for epoch in range(epochs):
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

    # record the loss and learning rate
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

    if (epoch + 1) % 10 == 0:
        mask_list = []
        for index in range(1):
            mask, p = dynamics.compute_masks(
                torch.tanh(output[index, 1:3, :, :]).detach().cpu().numpy(),
                torch.sigmoid(output[index, 0, :, :]).detach().cpu().numpy(),
                flow_threshold=0.0,
                min_size=5,
            )
            mask_list.append(mask)
        mask_list = np.array(mask_list, dtype=np.float32)
        mask = torch.tensor(mask_list, dtype=torch.float32).unsqueeze(1)

        writer.add_images("predict_image/mask", mask, epoch)
        writer.add_images("image/input", image[:4, :, :, :], epoch)
        writer.add_images(
            "predict_image/probability",
            torch.where(torch.sigmoid(output[:4, 0:1, :, :]) > 0.5, 1.0, 0),
            epoch,
        )
        writer.add_images("gt_image/probability", flow[:4, 0:1, :, :], epoch)

        writer.add_images(
            "predict_image/y_flow", (torch.tanh(output[:4, 1:2, :, :]) + 1) / 2, epoch
        )
        writer.add_images(
            "predict_image/x_flow", (torch.tanh(output[:4, 2:3, :, :]) + 1) / 2, epoch
        )
        writer.add_images("gt_image/y_flow", (flow[:4, 1:2, :, :] + 1) / 2, epoch)
        writer.add_images("gt_image/x_flow", (flow[:4, 2:3, :, :] + 1) / 2, epoch)

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f"./checkpoint/U_Mamba/model_{epoch+1}.pth")

writer.close()
