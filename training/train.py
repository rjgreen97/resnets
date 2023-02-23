import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from training.training_config import TrainingConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)

config = TrainingConfig()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.49139968, 0.48215841, 0.44653091),
            std=(0.24703223, 0.24348513, 0.26158784),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)
train_dataloader = torch.utils.data.Dataloader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)
test_dataloader = torch.utils.data.Dataloader(
    test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
