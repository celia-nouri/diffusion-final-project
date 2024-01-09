from unets import *
from ddpm import *
from improved_ddpm import *
import os
import math
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim


def train(args):
    device = args["device"]
    train_dataloader, val_dataset = get_data(args["dataset_path"], args["train_folder"], args["val_folder"],
                                             args["batch_size"], args["num_workers"])
    model = UNetModule().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args["lr"])

    # define the diffusion module according to the model_type
    diffusion = Diffusion(img_size=args["img_size"], device=device)
    if args["model_type"] == "improved_ddpm":
        diffusion = ImprovedDiffusion(img_size=args["img_size"], noise_schedule=args["noise_schedule"], learn_var=args["learn_var"], loss_type=args["loss_type"], device=device)

    logger = SummaryWriter(os.path.join("runs", args["run_name"]))
    l = len(train_dataloader)
    epochs = args["epochs"]
    print(f"Train DDPM model for {epochs} epochs...")
    if not os.path.exists(args["run_name"]):
        # Create the directory if it doesn't exist
        os.makedirs(args["run_name"])

    for epoch in range(args["epochs"]):
        print(f"Starting epoch number {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, (images, _) in enumerate(pbar):
            # images is [batch_size, 3, H:img_size, W:img_size]
            images = images.to(device)
            # TODO(celia): add option to use importance sampling schedule to decide on timestamp t
            # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            t = diffusion.get_n_timesteps(images.shape[0]).to(device)
            xt, noise = diffusion.q_sample(images, t)
            model_output = model(xt, t)
            loss_terms = diffusion.training_losses(model_output, noise, t, images, xt)
            loss = loss_terms["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("LOSS " + diffusion.loss_type, loss.item(), global_step=epoch * l + i)
        print(f"After epoch {epoch}, training loss is {loss}")
        sampled_images = diffusion.generate_images(model, n=images.shape[0])
        path = os.path.join("results", args["run_name"] + 'test')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Use the trained model to generate
        for i in range(sampled_images.shape[0]):
            # Normalize the tensor values to the range [0, 1]
            normalized_image = sampled_images[i, :, :, :] / 255.0
            uint8_image_normalized = (normalized_image * 255.0).clamp_(0, 255).to(torch.uint8)
            reshaped_image = np.transpose(uint8_image_normalized, (1, 2, 0))
            reshaped_image_np = reshaped_image.to(torch.uint8).cpu().numpy()
            image = Image.fromarray(reshaped_image_np)
            image.save('{}'.format(epoch) + '_' + '{}.png'.format(i))
    torch.save(model.state_dict(), os.path.join(args["run_name"], f"ckpt.pt"))
