import torch
from torch import Tensor, nn

from asg.losses import EmbeddingReconLoss, VAEKLLoss
from asg.models.base import BaseDACModel, ForwardExtras, LossDict

NUM_LAYERS = 1


class Model2(BaseDACModel):
    def __init__(self, T: int = 375, h_dim: int = 1024 * 1) -> None:
        super().__init__()

        self.h_dim = h_dim

        self.encoder = Model2Encoder(T, self.z_dim, h_dim)
        self.decoder = Model2Decoder(T, h_dim, self.z_dim)

        self.recon_loss_fn = EmbeddingReconLoss()
        self.kl_loss_fn = VAEKLLoss()

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, ForwardExtras]:
        """
        Args:
            z: [B, T, z_dim]

        Returns:
            Tuple of
            mu: [B, h_dim]
            log_var: [B, h_dim]
            z_hat: [B, T, z_dim]
        """

        mu, log_var = self.encoder(z)
        h = self.reparameterize(mu, log_var)
        z_hat = self.decoder(h)

        return z_hat, dict(mu=mu, log_var=log_var)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
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

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S]

        Returns:
            Tuple of
            mu: [B, h_dim]
            log_var: [B, h_dim]
            z: [B, T, z_dim]
        """

        z = self.samples_to_z(x)
        mu, log_var = self.encoder(z)

        return mu, log_var, z

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, h_dim]

        Returns:
            y: [B, S]
        """

        with torch.no_grad():
            z_hat = self.decoder(h)
            return self.z_to_samples(z_hat)

    def get_loss(
        self, y_true: Tensor, y_pred: Tensor, extras: ForwardExtras
    ) -> tuple[Tensor, LossDict]:
        recon_loss = self.recon_loss_fn(y_true, y_pred)

        mu = extras["mu"]
        log_var = extras["log_var"]
        kl_loss = self.kl_loss_fn(mu, log_var)

        loss = (recon_loss + 0.0001 * kl_loss).mean()

        return loss, dict(
            loss=loss.item(),
            recon_loss=recon_loss.mean().item(),
            kl_loss=kl_loss.mean().item(),
        )


class Model2Encoder(nn.Module):
    def __init__(
        self,
        T: int,
        in_dim: int,
        out_dim: int,
        mid_dim: int = 1024,
    ):
        super().__init__()

        self.T = T
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
        )

        self.pos_encodings = nn.Parameter(torch.randn(1, T, mid_dim) * 0.02)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=mid_dim,
                nhead=mid_dim // 16,
                dim_feedforward=mid_dim * 2,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=NUM_LAYERS,
            enable_nested_tensor=False,
        )

        self.pre_out_proj = nn.Sequential(
            nn.LeakyReLU(),
            nn.LayerNorm(mid_dim),
        )

        self.out_proj_mu = nn.Linear(mid_dim, out_dim)
        self.out_proj_log_var = nn.Linear(mid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B x T x in_dim]

        Returns:
            mu: [B x out_dim]
            log_var: [B x out_dim]
        """

        B, T, _ = x.shape
        assert x.shape == (B, T, self.in_dim), x.shape

        x = self.in_proj(x)
        assert x.shape == (B, T, self.mid_dim)

        x = x + self.pos_encodings
        assert x.shape == (B, T, self.mid_dim)

        x = self.transformer(x)
        assert x.shape == (B, T, self.mid_dim)

        x = x.mean(dim=1)
        assert x.shape == (B, self.mid_dim)

        x = self.pre_out_proj(x)
        assert x.shape == (B, self.mid_dim)

        mu = self.out_proj_mu(x)
        assert mu.shape == (B, self.out_dim)

        log_var = self.out_proj_log_var(x)
        assert log_var.shape == (B, self.out_dim)

        return mu, log_var


class Model2Decoder(nn.Module):
    def __init__(
        self,
        T: int,
        in_dim: int,
        out_dim: int,
        mid_dim: int = 1024,
    ):
        super().__init__()

        self.T = T
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.in_memory_proj = nn.Linear(in_dim, mid_dim)
        self.in_token_proj = nn.Linear(in_dim, mid_dim)

        self.pos_encodings = nn.Parameter(torch.randn(1, T, mid_dim) * 0.02)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=mid_dim,
                nhead=mid_dim // 16,
                dim_feedforward=mid_dim * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=NUM_LAYERS,
        )

        self.out_proj = nn.Sequential(
            nn.LeakyReLU(),
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B x in_dim] e.g. [1 x 64]

        Returns:
            embedding: [B x T x out_dim]
        """

        B, in_dim = h.shape
        T = self.T

        h = h.unsqueeze(1)
        assert h.shape == (B, 1, self.in_dim)

        tokens = self.in_token_proj(h)
        assert tokens.shape == (B, 1, self.mid_dim)

        tokens = h + self.pos_encodings
        assert tokens.shape == (B, T, self.mid_dim)

        memory = self.in_memory_proj(h)
        assert memory.shape == (B, 1, self.mid_dim)

        x = self.transformer(
            tgt=tokens,
            memory=memory,
            # mask=self.causal_mask,
        )
        assert x.shape == (B, T, self.mid_dim)

        x = self.out_proj(x)
        assert x.shape == (B, T, self.out_dim)

        return x


def verify_batch_independence(model, inputs, criterion=None):
    """
    Verify that batch elements don't influence each other's gradients.

    Args:
        model: your nn.Module
        inputs: a batch tensor, shape (B, ...)
        criterion: optional loss fn; defaults to output.sum()
    """
    B = inputs.shape[0]
    assert B >= 2, "Need at least 2 batch elements to test independence"

    # --- Pass 1: full batch, but zero out all losses except example 0 ---
    model.zero_grad()
    z_hat, extras = model(inputs)  # (B, ...)
    out = z_hat

    # Build a mask that keeps only example 0's contribution to the loss
    mask = torch.zeros_like(out)
    mask[0] = 1.0
    loss = (out * mask).sum()  # or criterion(out * mask, targets * mask)
    loss.backward()

    # Snapshot gradients from the masked-batch pass
    grads_masked = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    # --- Pass 2: single example only (the ground truth) ---
    model.zero_grad()
    z_hat, extras = model(inputs[[0]])  # batch of size 1
    out_single = z_hat
    loss_single = out_single.sum()
    loss_single.backward()

    grads_single = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    # --- Compare ---
    print("Checking batch independence...\n")
    all_ok = True
    for name in grads_single:
        g_masked = grads_masked[name]
        g_single = grads_single[name]
        max_diff = (g_masked - g_single).abs().max().item()
        ok = max_diff < 1e-5
        status = "✓" if ok else "✗  <-- PROBLEM"
        print(f"  {name:50s}  max_diff={max_diff:.2e}  {status}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("✓ All gradients match — batch elements are independent.")
    else:
        print("✗ Gradient mismatch detected — cross-batch contamination present!")

    return all_ok


def main():
    model = Model2()
    model.double()
    model.eval()

    B, T = 5, 375
    inputs = torch.randn(B, T, model.z_dim, dtype=torch.double)

    verify_batch_independence(model, inputs)


if __name__ == "__main__":
    main()
