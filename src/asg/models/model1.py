import torch
import torch.nn.functional as F
from torch import Tensor, nn

from asg.losses import EmbeddingReconLoss, VAEKLLoss
from asg.models.base import BaseDACModel, ForwardExtras, LossDict

NHEAD = 32
DIM_FEEDFORWARD = 1024 * 2
NUM_LAYERS = 1


class Model1(BaseDACModel):
    def __init__(self, h_dim: int = 128) -> None:
        super().__init__()

        self.h_dim = h_dim

        self.encoder = Model1Encoder(self.z_dim, h_dim)
        self.decoder = Model1Decoder(h_dim, self.z_dim)

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


class Model1Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mid_dim: int = 128,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
        )

        self.cls_token_count = 1
        self.cls_tokens = nn.Parameter(
            torch.randn(1, self.cls_token_count, mid_dim) * 0.02
        )

        self.pos_encodings = nn.Parameter(torch.randn(1, 375, mid_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=mid_dim,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=NUM_LAYERS,
            enable_nested_tensor=False,
        )

        self.pre_out_proj = nn.Sequential(
            nn.LayerNorm(self.cls_token_count * mid_dim),
            nn.ReLU(),
        )

        self.out_proj_mu = nn.Linear(self.cls_token_count * mid_dim, out_dim)
        self.out_proj_log_var = nn.Linear(self.cls_token_count * mid_dim, out_dim)

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

        cls_tokens = self.cls_tokens.expand(B, -1, -1)

        x = x + self.pos_encodings

        x = torch.cat([cls_tokens, x], dim=1)
        assert x.shape == (B, self.cls_token_count + T, self.mid_dim)

        x = self.transformer(x)
        assert x.shape == (B, self.cls_token_count + T, self.mid_dim)

        x = x[:, 0 : self.cls_token_count, :]  # CLS token outputs
        assert x.shape == (B, self.cls_token_count, self.mid_dim)

        x = x.flatten(start_dim=1)
        assert x.shape == (B, self.cls_token_count * self.mid_dim)

        x = self.pre_out_proj(x)

        mu = self.out_proj_mu(x)
        log_var = self.out_proj_log_var(x)
        assert mu.shape == (B, self.out_dim)
        assert log_var.shape == (B, self.out_dim)

        return mu, log_var


class Model1Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mid_dim: int = 128,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.in_proj = nn.Linear(in_dim, mid_dim)

        pos_dim = mid_dim
        # pos_dim = 128
        self.pos_encodings = nn.Parameter(torch.randn(1, 375, pos_dim) * 0.02)
        self.pos_proj = nn.Linear(pos_dim, mid_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=mid_dim,
            nhead=32,
            dim_feedforward=mid_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            layer,
            num_layers=NUM_LAYERS,
        )

        self.out_proj = nn.Linear(mid_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B x in_dim] e.g. [1 x 64]

        Returns:
            embedding: [B x T x out_dim]
        """

        B, in_dim = h.shape
        T = 375

        h = self.in_proj(h)
        h = h.unsqueeze(1)
        assert h.shape == (B, 1, self.mid_dim)

        # tokens = tokens.unsqueeze(1).expand(B, T, self.mid_dim)
        # tokens = tokens * self.pos_proj(self.pos_encodings)
        # tokens = self.pos_proj(self.pos_encodings)
        tokens = h * self.pos_proj(self.pos_encodings)
        # tokens = tokens.expand(B, T, self.mid_dim)
        assert tokens.shape == (B, T, self.mid_dim)

        x = self.transformer(
            tgt=tokens,
            memory=h,
            # mask=self.causal_mask,
        )
        assert x.shape == (B, T, self.mid_dim)

        x = self.out_proj(
            F.layer_norm(
                F.relu(x),
                (self.mid_dim,),
            ),
        )
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
    mu, log_var, z_hat = model(inputs)  # (B, ...)
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
    mu, log_var, z_hat = model(inputs[[0]])  # batch of size 1
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
    model = Model1()
    model.double()
    model.eval()

    B, T = 5, 375
    inputs = torch.randn(B, T, model.z_dim, dtype=torch.double)

    verify_batch_independence(model, inputs)


if __name__ == "__main__":
    main()
