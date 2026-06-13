from abc import ABC, abstractmethod

from torch import Tensor, nn

LossDict = dict[str, float]


class Loss(ABC):
    @abstractmethod
    def __call__(self, *args) -> Tensor:
        """Returns a loss tensor of shape (B,)."""
        ...


class EmbeddingReconLoss(Loss):
    def __init__(self):
        self._loss_fn = nn.MSELoss(reduction="none")

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        assert input.dim() >= 2

        B = input.shape[0]

        loss = self._loss_fn(input, target)
        loss = loss.reshape(B, -1)
        loss = loss.mean(dim=1)

        assert loss.shape == (B,)
        return loss


class VAEKLLoss(Loss):
    def __call__(self, mu: Tensor, log_std: Tensor) -> Tensor:
        """"""

        assert mu.shape == log_std.shape
        assert mu.dim() in (2, 3)

        B = mu.shape[0]

        kl_loss = -0.5 * (1 + log_std - mu.pow(2) - log_std.exp()).sum(dim=-1)

        if kl_loss.dim() == 2:
            kl_loss = kl_loss.mean(dim=1)

        assert kl_loss.shape == (B,)
        return kl_loss
