import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from cellpose import dynamics

from model import UNet
from data import CellFlowDataset

# 设置 TensorBoard
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 数据转换
transform = transforms.Compose(
    [
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((1024, 1024)),
    ]
)

# 加载数据集
dataset = CellFlowDataset(
    "/data/disk01/cell/cell_1360x1024/annotation/train.json",
    "/data/disk01/cell/cell_1360x1024/train",
    transform=transform,
)

dataloaders = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# 模型
model = UNet().to(device)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001
)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)


# 训练
epochs = 1000
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    print("-" * 10)
    model.train()

    running_loss = 0.0
    mask_loss = 0.0
    flow_loss = 0.0

    for i, (image, flow) in enumerate(dataloaders):
        image = image.to(device)
        flow = flow.to(device)

        optimizer.zero_grad()
        output = model(image)

        loss1 = nn.BCEWithLogitsLoss()(output[:, 0:1, :, :], flow[:, 0:1, :, :])
        # loss2 = nn.MSELoss()(torch.tanh(output[:,1:3,:,:]), flow[:,1:3,:,:])
        loss2 = torch.mean(
            torch.abs(torch.tanh(output[:, 1:3, :, :]) - flow[:, 1:3, :, :])
        )
        loss = 0.2*loss1 + 0.8*loss2

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        mask_loss += loss1.item()
        flow_loss += loss2.item()

    scheduler.step()
    average_loss = running_loss / len(dataloaders)
    mask_loss = mask_loss / len(dataloaders)
    flow_loss = flow_loss / len(dataloaders)

    print(
        f"Epoch {epoch+1}, loss: {average_loss}, mask_loss: {mask_loss}, flow_loss: {flow_loss}"
    )
    writer.add_scalar("learning rate", optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("loss/all average loss", average_loss, epoch)
    writer.add_scalar("loss/mask loss", mask_loss, epoch)
    writer.add_scalar("loss/flow loss", flow_loss, epoch)

    running_loss = 0.0
    mask_loss = 0.0
    flow_loss = 0.0


    if (epoch + 1) % 10 == 0:
        mask_list=[]
        for index in range(2):
            mask, p = dynamics.compute_masks(flow[index,1:,:,:].cpu().numpy(), flow[index,0,:,:].cpu().numpy(), flow_threshold=0.0,min_size=5)
            mask_list.append(mask)
        mask=torch.tensor(mask_list,dtype=torch.float32).unsqueeze(1)
        writer.add_images(f"predict_image/mask", mask, epoch)
        writer.add_images("image/input", image[:4, :, :, :], epoch)
        writer.add_images(
            "predict_image/probability",
            torch.where(nn.Sigmoid()(output[:4, 0:1, :, :]) > 0.5, 1.0, 0),
            epoch,
        )  # 阈值化
        writer.add_images("gt_image/probability", flow[:4, 0:1, :, :], epoch)

        writer.add_images(
            "predict_image/y_flow", (torch.tanh(output[:4, 1:2, :, :]) + 1) / 2, epoch
        )
        writer.add_images(
            "predict_image/x_flow", (torch.tanh(output[:4, 2:3, :, :]) + 1) / 2, epoch
        )
        writer.add_images("gt_image/y_flow", (flow[:4, 1:2, :, :] + 1) / 2, epoch)
        writer.add_images("gt_image/x_flow", (flow[:4, 2:3, :, :] + 1) / 2, epoch)


    if (epoch + 1) % 50 == 0:  # 每50个epoch保存一次模型
        torch.save(model, f"model_{epoch+1}.pth")

# 关闭 TensorBoard 写入器
writer.close()
