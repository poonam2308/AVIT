# datasets.py
from __future__ import annotations
from typing import Tuple, Optional
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def cifar10_loaders(data_root: str, batch_size: int, num_workers: int = 4, img_size: int = 224):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=8),  # optional but helps
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(root=data_root, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def cifar10_loaders_old(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return trainloader, testloader


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)



def imagenet_style_loaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
):
    """
    Works for:
      Imagenette
      Imagewoof
      Tiny-ImageNet
      Custom ImageFolder datasets
    """
    train_dir = f"{data_root}/train"
    val_dir = f"{data_root}/val"

    transform_train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_val = T.Compose([
        T.Resize(int(img_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    valset = torchvision.datasets.ImageFolder(val_dir, transform=transform_val)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valloader = DataLoader(
        valset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, valloader