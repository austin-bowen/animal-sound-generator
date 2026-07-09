import math
from abc import ABC, abstractmethod

import dac
import torch
from torch import Tensor, nn

from asg.utils import set_requires_grad

ForwardExtras = dict[str, Tensor]
LossDict = dict[str, float]


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args) -> tuple[Tensor, ForwardExtras]: ...

    @abstractmethod
    def get_loss(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        extras: ForwardExtras,
    ) -> tuple[Tensor, LossDict]: ...


class BaseDACModel(BaseModel, ABC):
    def __init__(
        self,
        model_type: str = "24khz",
        z_dim: int = 1024,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim

        self.dac_z_std = 1.0

        # Load pretrained model
        model_path = dac.utils.download(model_type=model_type)
        dac_model = dac.DAC.load(model_path)
        dac_model.eval()

        set_requires_grad(dac_model, False)

        self.dac_model = dac_model

    def samples_to_z(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S]

        Returns:
            z: [B, T, z_dim]
        """

        B, _ = x.shape

        with torch.no_grad():
            # Preprocess: resample + pad to model's expected hop size
            x = x.unsqueeze(1)
            x = self.dac_model.preprocess(x, sample_rate=None)

            z = self.dac_model.encoder(x)
            T = z.shape[-1]
            assert z.shape == (B, self.z_dim, T)

            z = z.permute(0, 2, 1)
            assert z.shape == (B, T, self.z_dim)

            return z / self.dac_z_std

    def z_to_samples(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, T, z_dim]

        Returns:
            y: [B, S]
        """

        B, T, z_dim = z.shape
        assert z_dim == self.z_dim

        with torch.no_grad():
            z = z.permute(0, 2, 1)
            assert z.shape == (B, self.z_dim, T)

            z = z * self.dac_z_std
            y = self.dac_model.decoder(z)  # [B, 1, S]  reconstructed waveform

            y = y.squeeze(1)
            _, S = y.shape
            assert y.shape == (B, S)

            return y


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()

        # Precompute the encoding table once, at construction time
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(
            1
        )  # (max_len, 1)

        # div_term = 1 / (10000^(2i/d_model)), computed in log-space for stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for easy broadcasting over batch

        # Register as a buffer: moves with .to(device), saved in state_dict (optionally),
        # but NOT a learnable parameter and not touched by the optimizer
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
