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

def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def untar_data(file_path='./datasets/archive/cifar10.tgz', extract_path="./data/cifar10.tgz"):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(extract_path)


"""
def get_cifar(cifar100=False, img_size=64):
    "Download and extract CIFAR"
    cifar10_url = 'https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz'
    cifar100_url = 'https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz'
    if img_size==32:
        return untar_data()
    else:
        get_kaggle_dataset("datasets/cifar10_64", "joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution")
        return Path("datasets/cifar10_64/cifar10-64")
"""

def one_batch(dl):
    return next(iter(dl))


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(data_path, train_folder, val_folder, batch_size=25, num_workers=2, slice_size=1):
    #train_transforms = torchvision.transforms.Compose([
        # T.Resize(img_size + int(.25*img_size)),  # args.img_size + 1/4 *args.img_size
        # T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        # T.ToTensor(),
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #])

    transform = torchvision.transforms.Compose([
        #T.Resize(img_size),
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

def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def extract_into_tensor(arr, timesteps_indices, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
