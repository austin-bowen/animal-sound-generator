import torch
from torch import Tensor, nn

from asg.models.base import BaseDACModel, ForwardExtras, LossDict


class Memorizer(BaseDACModel):
    def __init__(
        self,
        # T: int = 375,
        T: int = 150,
        h_dim: int = 1024,
    ) -> None:
        super().__init__()

        self.h_dim = h_dim

        self.embeddings = nn.Parameter(torch.randn(1, T, h_dim) * 1)
        # init.kaiming_uniform_(self.embeddings, a=math.sqrt(5))
        self.scale = nn.Parameter(torch.ones(1, T, 1))
        self.bias = nn.Parameter(torch.zeros(1, T, 1))
        self.out_proj = nn.Linear(h_dim, 1024)
        self.norm = nn.RMSNorm(h_dim)

        # self.recon_loss_fn = EmbeddingReconLoss()
        self.recon_loss_fn = nn.SmoothL1Loss(reduction="none")

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, ForwardExtras]:
        """
        Args:
            z: [B, T, z_dim]

        Returns:
            z_hat: [B, T, z_dim]
            extras: Dict of additional outputs
        """

        B = z.size(0)

        z_hat = self.embeddings.expand(B, -1, -1)
        # z_hat = z_hat * self.scale + self.bias
        # z_hat = self.norm(z_hat)
        z_hat = self.out_proj(z_hat)

        return z_hat, dict()

    def get_loss(
        self, y_true: Tensor, y_pred: Tensor, extras: ForwardExtras
    ) -> tuple[Tensor, LossDict]:
        loss = self.recon_loss_fn(y_true, y_pred).mean()

        return loss, dict(
            loss=loss.item(),
        )
