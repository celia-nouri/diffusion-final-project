
import utils
from utils import *
from train import *
from unets import UNetModule
from torch.utils.tensorboard import SummaryWriter
import torchvision

"""
    Implementation of DDPM paper (https://arxiv.org/abs/2006.11239) from scratch.
"""


class Diffusion:
    """
        The diffusion module handles the noising and denoising operations. The implementation is based on the DDPM paper cited above.
    """

    def __init__(self, img_size=32, T=1000, beta_1=1e-4, beta_T=0.02, device="cpu"):
        self.img_size = img_size
        self.device = device
        self.loss_type = "mse"

        # number of noising steps
        self.T = T

        # betas are the forward process variances. In DDPM paper, they are fixed to constants.
        self.beta_1 = beta_1
        self.beta_T = beta_T

        # DDPM paper assigns beta values according to a linear noising schedule, use improved_ddpm for cosine schedule
        self.beta = torch.linspace(self.beta_1, self.beta_T, self.T)  # [T]

        # alpha, alpha_bar, sqrt_alpha_bar and sqrt_one_minus_alpha_bar are defined to ease notations
        self.alpha = 1. - self.beta  # [T]
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # [T]
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)  # [T]
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)  # [T]

    def q_sample(self, x, t):
        """
        q sample preforms the forward noising process on a batch of images.
        It computes the posterior: q(xt/x0) = N(xt; sqrt(alpha_bar) * x_0, (1- alpha_bar)*I)
        :param x: images to noise
        :param t: timestamp of images to be noised
        :return: the noised image after t noising steps, the random noise applied
        """

        mean = self.sqrt_alpha_bar[t][:, None, None, None] * x
        noise = torch.randn_like(x)
        std = self.sqrt_one_minus_alpha_bar[t][:, None, None, None] * noise
        return mean + std, noise

    def get_n_timesteps(self, n):
        """
        get n timesteps samples n timesteps from 1 to T, at random.
        :param n: Number of random timesteps to return
        :return: tensor of n timesteps
        """
        return torch.randint(1, self.T, (n,))

    # reverse process, p_theta.
    def generate_images(self, model, n):
        """
        Generate images implements the sampling algorithm (Algorithm 2 of the DDPM paper). The sampling algorithm samples for each image, random noise from a centered normal distribution and denoises it T times, following the model noise prediction.
        :param model: parametrized model used to sample images
        :param n: number of images to sample
        :return: x0, the sampled image denoised using the model.
        """
        print(f"Generate {n} images")
        model.eval()
        with torch.no_grad():
            # start from random noise
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.T)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                # use the model to predict noise to be removed
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    z = torch.randn_like(x)
                # don't add noise variance for first timestamp
                else:
                    z = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / sqrt_one_minus_alpha_bar * predicted_noise) + torch.sqrt(
                    beta) * z
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def training_losses(self, model_output, noise, t, x0, xt):
        """
        Compute training losses for a single timestep.

        :param model_output: the model to evaluate loss on.
        :param x0: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param xt:
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        terms = {}
        if self.loss_type == "mse":
            mse = nn.MSELoss()
            terms["loss"] = mse(noise, model_output)
        return terms
