import torch
import torch.nn as nn
from torch import Tensor

from asg.losses import EmbeddingReconLoss
from asg.models.base import BaseDACModel, ForwardExtras, LossDict


class ZELSA(BaseDACModel):
    def __init__(
        self,
        h_dim: int = 64,
    ) -> None:
        super().__init__()

        self.h_dim = h_dim

        self.matrix = nn.Parameter(torch.randn(self.z_dim, h_dim) * 0.01)
        self.norm = nn.LayerNorm(self.h_dim, elementwise_affine=False, bias=False)

        self.loss_fn = EmbeddingReconLoss()

    def forward(self, z: Tensor) -> tuple[Tensor, ForwardExtras]:
        """
        Args:
             z: [B, T, z_dim]
        """

        B, T, z_dim = z.shape
        assert z_dim == self.z_dim

        h = z @ self.matrix
        h = self.norm(h)
        assert h.shape == (B, T, self.h_dim)

        z_hat = h @ self.matrix.T
        assert z_hat.shape == (B, T, self.z_dim)

        return z_hat, dict(h=h)

    def get_loss(
        self, y_true: Tensor, y_pred: Tensor, extras: ForwardExtras
    ) -> tuple[Tensor, LossDict]:
        loss = self.loss_fn(y_true, y_pred).mean()
        return loss, dict(loss=loss.item())
