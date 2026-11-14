import torch
import torch.nn.functional as F
from torch import nn

from .diffusion import BernoulliDiffusion, NoiseConditionedMLP


class MlpDiffusionFingerprintHead(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor,
        d_model: int,
        fingerprint_dim: int,
        hidden_dim: int,
        diffusion_steps: int = 100,
        s: float = 0.01,
    ):
        super().__init__()
        self.var_sched = BernoulliDiffusion(num_steps=diffusion_steps, s=s)
        self.denoiser = NoiseConditionedMLP(
            dim_in=d_model + fingerprint_dim,
            dim_out=fingerprint_dim,
            dim_hidden=hidden_dim,
        )
        self.fingerprint_dim = fingerprint_dim
        self.register_buffer("fp_pos_weight", pos_weight)

    def forward(self, h: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        combined = torch.cat([h, x_t], dim=-1)
        return self.denoiser(combined, t, self.var_sched)

    def get_loss(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        num_timestep_samples: int = 32,
    ):
        h = h[None].repeat(num_timestep_samples, 1, 1)  # (n_smp, bsz, d_model)
        fp_gt = fp_target.repeat(num_timestep_samples, 1, 1)  # (n_smp, bsz, fp_dim)

        timestep = torch.randint(
            low=1,
            high=self.var_sched.num_steps + 1,
            size=fp_gt.shape[:-1],  # (n_smp, bsz)
            device=h.device,
        )
        noisy_sample = self.var_sched.add_noise(fp_gt, timestep.unsqueeze(-1))
        noisy_fp = noisy_sample.value

        p_denoised = torch.sigmoid(
            self.forward(
                h=h.view(-1, h.size(-1)),  # (n_smp * bsz, d_model)
                x_t=noisy_fp.view(-1, self.fingerprint_dim),
                t=timestep.flatten(),
            )
        ).view_as(noisy_fp)

        posterior_gt = self.var_sched.posterior(p_t=noisy_fp, p_0=fp_gt, t=timestep)
        posterior_pred = self.var_sched.posterior(
            p_t=noisy_fp, p_0=p_denoised, t=timestep
        )

        loss_diffusion = F.binary_cross_entropy(
            input=posterior_pred, target=posterior_gt, reduction="mean"
        )
        loss_bce = F.binary_cross_entropy(
            input=p_denoised, target=fp_gt, reduction="mean"
        )
        return loss_diffusion * 0.2 + loss_bce * 0.8

    def predict_probability(self, h: torch.Tensor, x_t: torch.Tensor = None):
        bsz = h.shape[0]
        fp_shape = (bsz, self.fingerprint_dim)
        if x_t is None:
            x_t = torch.randint(0, 2, fp_shape, device=h.device).float()

        final_p_0_pred = None
        for step in reversed(range(1, self.var_sched.num_steps + 1)):
            t_tensor = torch.full((bsz,), step, device=h.device)
            logits = self.forward(h, x_t, t_tensor)
            p_0_pred = torch.sigmoid(logits)
            final_p_0_pred = p_0_pred

            denoise_result = self.var_sched.denoise(
                x_t=x_t, p_0_pred=p_0_pred, t=t_tensor
            )
            x_t = denoise_result.value
        return final_p_0_pred
