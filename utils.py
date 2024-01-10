import os, random
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import tarfile

cifar_labels = "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck".split(",")

# Running DDPM with MSE loss by default
default_args = {
    "run_name": "test_run",
    "epochs": 100,
    "batch_size": 12,
    "img_size": 32,
    "dataset_path": "./datasets/data/cifar10",
    "device": "cpu",
    "lr": 3e-4,
    "train_folder": "train",
    "val_folder": "test",
    "slice_size": 1,
    "num_workers": 2,
    "model_type": "ddpm",  # one of [ddpm, improved_ddpm, cond_ddpm]
    "loss_type": "mse",  # one of [mse, rescaled_mse, kl, rescaled_kl]
    "learn_var": False,  # cannot be True for ddpm model type
    "noise_schedule": "linear",  # one of [linear, cosine]
}


def untar_data(file_path='./datasets/archive/cifar10.tgz', extract_path="./data/cifar10.tgz"):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(extract_path)


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(data_path, train_folder, val_folder, batch_size=25, num_workers=2, slice_size=1):
    transform = torchvision.transforms.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path, train_folder), transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path, val_folder), transform=transform)
    if slice_size > 1:
        train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), slice_size))
        val_dataset = torch.utils.data.Subset(val_dataset, indices=range(0, len(val_dataset), slice_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = DataLoader(val_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
    print(f"Created dataloaders for training (size is {len(train_dataset)/slice_size}) and validation (size is {len(val_dataset)/slice_size})")
    return train_dataloader, val_dataset
