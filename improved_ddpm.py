from ddpm import Diffusion
from losses import *
from train import *
import math
import numpy as np
import torch
import functools


class ImprovedDiffusion(Diffusion):
    def __init__(self, img_size=32, T=1000, beta_1=1e-4, beta_T=0.02, noise_schedule="linear", learn_var=True, loss_type="KL", device="cpu"):
        # Call the Diffusion __init__ method
        super().__init__(img_size, T, beta_1, beta_T, device)

        # true if the unet model learns gaussian means and variances, false if it only learns the mean and variances are fixed
        self.learn_var = learn_var

        # specifies which loss term should be used
        self.loss_type = loss_type

        # In improved DDPM, the cosine noise schedule is introduced to smooth out noise addition through the T noising steps
        assert noise_schedule in ["linear", "cosine"]
        if noise_schedule == "cosine":
            betas = []
            for i in range(T):
                t1 = i / T
                t2 = (i + 1) / T
                betas.append(min((1 - math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2) / (
                            math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2), 0.999))
            self.beta = torch.tensor(betas)  # [T]

        # other useful notations
        self.alpha_bar_prev = torch.cat((torch.tensor([1.0]), self.alpha_bar[:-1]))
        self.alpha_bar_next = torch.cat((self.alpha_bar[1:], torch.tensor([0.0])))
        self.sqrt_one_over_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_one_over_alpha_bar_minus_one = torch.sqrt(1.0 / self.alpha_bar - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat((torch.tensor([self.posterior_variance[1]]), self.posterior_variance[1:]))
        )
        self.posterior_mean_coef1 = (
            self.beta * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bar_prev)
            * torch.sqrt(self.alpha)
            / (1.0 - self.alpha_bar)
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = self.posterior_mean_coef1[t][:, None, None, None] * x_start + self.posterior_mean_coef2[t][:, None, None, None] * x_t
        posterior_variance = self.posterior_variance[t][:, None, None, None]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t][:, None, None, None]

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def x0_from_xt(self, xt, t, noise):
        """
        x0_from_xt computes x0 from xt by reversing Equation 4 used in q_sample. See Equation 11 from Improved DDPM.
        :param xt: the image at timestamp t
        :param t: the timestamp
        :param noise: the noise predicted from model
        :return: x0
        """
        assert xt.shape == noise.shape
        return self.sqrt_one_over_alpha_bar[t][:, None, None, None] * xt - self.sqrt_one_over_alpha_bar_minus_one[t][:, None, None, None] * noise

    def p_mean_variance(self, model_output, x, t, clip_denoised=True, learned_type="learned_range"):
        """
        The p_mean_variance method computes p(x_{t-1} / x_t)
        :param model_output: output of the model, can be the gaussian mean or the mean and variances.
        :param x:
        :param t:
        :param clip_denoised:
        :param learned_type:
        :return:
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)

        # (1) Compute the model variance
        if self.learn_var:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_v_values = torch.split(model_output, C, dim=1)
            if learned_type == "learned":
                model_log_variance = model_v_values
                model_variance = torch.exp(model_log_variance)
            else:
                # Equation 15 (improved DDPM)
                min_log = self.posterior_log_variance_clipped[t][:, None, None, None]
                max_log = torch.log(self.beta)[t][:, None, None, None]
                # The model_v_values is [-1, 1] for [min_var, max_var], bring model output v values to [0,1] range
                norm_v_values = (model_v_values + 1) / 2
                model_log_variance = norm_v_values * max_log + (1 - norm_v_values) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            assert model_output.shape == (B, C, *x.shape[2:])
            model_variance = self.posterior_variance[t][:, None, None, None]
            model_log_variance = self.posterior_log_variance_clipped[t][:, None, None, None]
        pred_x0 = self.x0_from_xt(x, t, model_output)
        # clip pred x0 in [-1, 1] - optional
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(pred_x0, x, t)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x0": pred_x0,
        }

    def p_sample(self, model, x, t):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        model.eval()
        # use the model to predict noise to be removed
        model_output = model(x, t)
        out = self.p_mean_variance(
            model_output,
            x,
            t,
            clip_denoised=True,
            learned_type="learned_range",
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x0": out["pred_x0"]}

    def _vb_terms_bpd(self, model, x0, xt, t, clip_denoised=True):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x0, xt, t)
        out = self.p_mean_variance(model, xt, t, clip_denoised)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = kl.mean(dim=list(range(1, len(kl.shape)))) / torch.log(torch.tensor([2.0]))

        L_0 = -discretized_gaussian_log_likelihood(x0, means=out["mean"], log_scales=0.5 * out["log_variance"])
        assert L_0.shape == x0.shape
        L_0 = L_0.mean(dim=list(range(1, len(L_0.shape)))) / torch.log(torch.tensor([2.0]))

        # At the first timestep return L_0,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0),  L_0, kl)
        return {"output": output, "pred_x0": out["pred_x0"]}

    def training_losses(self, model_output, noise, t, x0, xt):
        """
        Compute training losses for a single timestep.

        :param model_output: the model to evaluate loss on.
        :param x0: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        terms = {}

        if self.loss_type == "KL" or self.loss_type == "rescaled_KL":
            terms["loss"] = self._vb_terms_bpd(model_output, x0, xt, t, False)["output"]
            if self.loss_type == "rescaled_KL":
                terms["loss"] *= self.T
        elif self.loss_type == "mse" or self.loss_type == "rescaled_MSE":
            if self.learn_var:
                B, C = xt.shape[:2]
                assert model_output.shape == (B, C * 2, *xt.shape[2:])
                model_output_means, model_v_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output_means.detach(), model_v_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(model_output, x0, xt, t, False)["output"]
                if self.loss_type == "rescaled_MSE":
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.T / 1000.0
            diff_scared_true_pred = (noise - model_output_means) ** 2
            terms["mse"] = diff_scared_true_pred.mean(dim=list(range(1, len(diff_scared_true_pred.shape))))
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError("loss type is not implemented, should be one of KL, rescaled_KL, MSE, rescaled_MSE but got " + self.loss_type)
        loss = terms["loss"]
        return terms
