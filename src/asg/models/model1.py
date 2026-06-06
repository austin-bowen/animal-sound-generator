import dac
import torch
import torch.nn.functional as F
from torch import nn

DAC_Z_STD = 3.627
NHEAD = 32
DIM_FEEDFORWARD = 1024 * 2
NUM_LAYERS = 1


class Model1(nn.Module):
    def __init__(self, h_dim: int = 16) -> None:
        super().__init__()

        self.h_dim = h_dim

        # Load pretrained model
        model_path = dac.utils.download(model_type="24khz")
        dac_model = dac.DAC.load(model_path)
        dac_model.eval()

        for param in dac_model.parameters():
            param.requires_grad = False

        self.dac_model = dac_model

        dac_z_dim = 1024
        self.encoder = Model1Encoder(dac_z_dim, h_dim)
        self.decoder = Model1Decoder(h_dim, dac_z_dim)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, 1024, T]

        Returns:
            Tuple of
            mu: [B, h_dim]
            log_var: [B, h_dim]
            z_hat: [B, 1024, T]
        """

        mu, log_var = self.encoder(z)
        h = self.reparameterize(mu, log_var)
        z_hat = self.decoder(h)

        return mu, log_var, z_hat

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
            z: [B, 1024, T]
        """

        z = self.get_dac_z(x)
        mu, log_var = self.encoder(z)

        return mu, log_var, z

    def get_dac_z(self, x: torch.Tensor) -> torch.Tensor:
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
            return z / DAC_Z_STD

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

    def z_to_samples(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, 1024, T]

        Returns:
            y: [B, S]
        """

        with torch.no_grad():
            z = z * DAC_Z_STD
            y = self.dac_model.decoder(z)  # [B, 1, S]  reconstructed waveform

            return y.squeeze(1)


class Model1Encoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cls_token_count = 1
        self.cls_tokens = nn.Parameter(
            torch.randn(1, self.cls_token_count, in_dim) * 0.02
        )

        self.pos_encodings = nn.Parameter(torch.randn(1, 375, in_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
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

        self.out_proj_mu = nn.Linear(self.cls_token_count * in_dim, out_dim)
        self.out_proj_log_var = nn.Linear(self.cls_token_count * in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B x in_dim x T] e.g. [1 x 1024 x 375]

        Returns:
            mu: [B x out_dim]
            log_var: [B x out_dim]
        """

        B, _, T = x.shape

        x = x.permute(0, 2, 1)
        assert x.shape == (B, T, self.in_dim), x.shape

        cls_tokens = self.cls_tokens.expand(B, -1, -1)

        x = x + self.pos_encodings

        x = torch.cat([cls_tokens, x], dim=1)
        assert x.shape == (B, self.cls_token_count + T, self.in_dim)

        x = self.transformer(x)
        assert x.shape == (B, self.cls_token_count + T, self.in_dim)

        x = x[:, 0 : self.cls_token_count, :]  # CLS token outputs
        assert x.shape == (B, self.cls_token_count, self.in_dim)

        x = x.flatten(start_dim=1)
        assert x.shape == (B, self.cls_token_count * self.in_dim)

        x = F.relu(x)

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
        mid_dim: int = 64,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.in_proj = nn.Linear(in_dim, mid_dim)

        # pos_dim = mid_dim
        pos_dim = 64
        self.pos_encodings = nn.Parameter(torch.randn(1, 375, pos_dim) * 0.02)
        self.pos_proj = nn.Linear(pos_dim, mid_dim, bias=False)

        layer = nn.TransformerDecoderLayer(
            d_model=mid_dim,
            nhead=8,
            dim_feedforward=mid_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            layer,
            num_layers=1,
        )

        self.out_proj = nn.Linear(mid_dim, out_dim)
        self.out_scale = nn.Linear(mid_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B x in_dim] e.g. [1 x 64]

        Returns:
            embedding: [B x out_dim x 375]
        """

        B, in_dim = h.shape

        h = self.in_proj(h)
        h = h.unsqueeze(1)
        assert h.shape == (B, 1, self.mid_dim)

        # tokens = tokens.unsqueeze(1).expand(B, 375, self.mid_dim)
        # tokens = tokens * self.pos_proj(self.pos_encodings)
        # tokens = self.pos_proj(self.pos_encodings)
        tokens = h + self.pos_proj(self.pos_encodings)
        # tokens = tokens.expand(B, 375, self.mid_dim)
        assert tokens.shape == (B, 375, self.mid_dim)

        x = self.transformer(
            tgt=tokens,
            memory=h,
            # mask=self.causal_mask,
        )
        assert x.shape == (B, 375, self.mid_dim)

        out = self.out_proj(F.relu(F.layer_norm(x, (self.mid_dim,))))
        # out = F.normalize(out, dim=-1)
        assert out.shape == (B, 375, self.out_dim)

        # scalar = self.out_scale(x) + 1
        # assert scalar.shape == (B, 375, 1)
        #
        # x = out * scalar
        # assert x.shape == (B, 375, self.out_dim)

        x = out

        x = x.permute(0, 2, 1)
        assert x.shape == (B, self.out_dim, 375)

        return x
