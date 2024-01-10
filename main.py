from utils import *
from train import *


def train_ddpm():
    args = default_args
    args["run_name"] = "ddpm_uncond_1"
    args["epochs"] = 20
    args["batch_size"] = 64
    train(args)


def train_improved_ddpm():
    args = default_args
    args["run_name"] = "improved_ddpm_0"
    args["epochs"] = 20
    args["batch_size"] = 64
    args["model_type"] = "improved_ddpm"
    args["loss_type"] = "mse"
    args["learn_var"] = True
    args["noise_schedule"] = "cosine"
    train(args)


if __name__ == '__main__':
    train_improved_ddpm()
