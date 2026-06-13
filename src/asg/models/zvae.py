import dac
import torch
import torch.nn as nn
from torch import Tensor


class ZVAE(nn.Module):
    def __init__(
        self,
        z_dim: int = 1024,
        h_dim: int = 128,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim

        self.dac_z_std = 1.0

        # Load pretrained model
        model_path = dac.utils.download(model_type="24khz")
        dac_model = dac.DAC.load(model_path)
        dac_model.eval()

        for param in dac_model.parameters():
            param.requires_grad = False

        self.dac_model = dac_model

        self.h_mu = nn.Linear(z_dim, h_dim)
        self.h_log_var = nn.Linear(z_dim, h_dim)

        self.h_to_z = nn.Linear(h_dim, z_dim)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, z_dim, T = z.shape
        assert z_dim == self.z_dim

        z = z.permute(0, 2, 1)
        assert z.shape == (B, T, self.z_dim)

        mu = self.h_mu(z)
        assert mu.shape == (B, T, self.h_dim)

        log_var = self.h_log_var(z)
        assert log_var.shape == (B, T, self.h_dim)

        h = self.reparameterize(mu, log_var)
        assert h.shape == (B, T, self.h_dim)

        z_hat = self.h_to_z(h)
        assert z_hat.shape == (B, T, self.z_dim)

        z_hat = z_hat.permute(0, 2, 1)
        assert z_hat.shape == (B, self.z_dim, T)

        return mu, log_var, z_hat

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> torch.Tensor:
        """
        Args:
            mu: [B, h_dim]
            log_var: [B, h_dim]

        Returns:
            h: [B, h_dim]
        """

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def samples_to_z(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S]

        Returns:
            z: [B, 1024, T]
        """

        with torch.no_grad():
            # Preprocess: resample + pad to model's expected hop size
            x = x.unsqueeze(1)
            x = self.dac_model.preprocess(x, sample_rate=None)

            z = self.dac_model.encoder(x).detach()
            return z / self.dac_z_std

    def z_to_samples(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, 1024, T]

        Returns:
            y: [B, S]
        """

        with torch.no_grad():
            z = z * self.dac_z_std
            y = self.dac_model.decoder(z)  # [B, 1, S]  reconstructed waveform

            return y.squeeze(1)
