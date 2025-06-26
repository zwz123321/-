import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def calculate_mean_std(dataset, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    mean = 0.
    std = 0.
    total_samples = 0

    for images, _ in loader:
        images = images.to(device)
        batch_size_images = images.size(0)
        images = images.view(batch_size_images, images.size(1), -1)  # shape: [B, C, H*W]

        mean += images.mean(dim=2).sum(dim=0)
        std += images.std(dim=2).sum(dim=0)
        total_samples += batch_size_images

    mean /= total_samples
    std /= total_samples

    return mean.cpu().numpy(), std.cpu().numpy()


if __name__ == "__main__":
    # 设置统一图像大小，比如 224x224
    target_size = 224

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),  # 确保是 RGB 图像
        transforms.Resize(target_size),                      # 缩放短边到 target_size
        transforms.CenterCrop(target_size),                  # 中心裁剪为 target_size x target_size
        transforms.ToTensor()
    ])

    dataset_root = 'GoogLeNet_1/data/train'
    train_dataset = datasets.ImageFolder(root=dataset_root, transform=transform)

    print("开始计算数据集的均值和标准差...")
    mean, std = calculate_mean_std(train_dataset)

    print("计算完成！")
    print("均值 Mean:", mean)
    print("标准差 Std:", std)