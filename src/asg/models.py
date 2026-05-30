import dac
import torch
import torch.nn.functional as F
from torch import nn


class Model0(nn.Module):
    def __init__(self, h_dim: int = 1024 * 8) -> None:
        super().__init__()

        # Load pretrained model
        model_path = dac.utils.download(model_type="24khz")
        dac_model = dac.DAC.load(model_path)
        dac_model.eval()

        for param in dac_model.parameters():
            param.requires_grad = False

        self.dac_model = dac_model

        dac_z_dim = 1024
        self.encoder = Model0Encoder(dac_z_dim, h_dim)
        self.decoder = Model0Decoder(h_dim, dac_z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S]

        Returns:
            Tuple of
            h: [B, h_dim]
            z_hat: [B, 1024, T]
            z: [B, 1024, T]
        """

        h, z = self.encode(x)
        z_hat = self.decoder(h)

        return h, z_hat, z

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S]

        Returns:
            Tuple of
            h: [B, h_dim]
            z: [B, 1024, T]
        """

        with torch.no_grad():
            # Preprocess: resample + pad to model's expected hop size
            x = x.unsqueeze(1)
            x = self.dac_model.preprocess(x, sample_rate=None)

            # z, codes, latents, commitment_loss, codebook_loss = self.dac_model.encode(x)
            z = self.dac_model.encoder(x).detach()

        h = self.encoder(z)

        return h, z

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, h_dim]

        Returns:
            y: [B, S]
        """

        z_hat = self.decoder(h)

        with torch.no_grad():
            y = self.dac_model.decoder(z_hat)  # [B, 1, S]  reconstructed waveform

        return y.squeeze(1)


class Model0Encoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cls_token_count = 8
        self.cls_tokens = nn.Parameter(torch.randn(1, self.cls_token_count, in_dim) * 0.02)

        self.pos_encodings = nn.Parameter(torch.randn(1, 375, in_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=8,
            dim_feedforward=in_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=1,
        )

        self.out_proj = nn.Linear(self.cls_token_count * in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B x in_dim x T] e.g. [1 x 1024 x 375]

        Returns:
            embedding: [B x out_dim]
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

        x = x[:, 0:self.cls_token_count, :]  # CLS token outputs
        assert x.shape == (B, self.cls_token_count, self.in_dim)

        x = x.flatten(start_dim=1)
        assert x.shape == (B, self.cls_token_count * self.in_dim)

        x = F.relu(x)

        x = self.out_proj(x)
        assert x.shape == (B, self.out_dim)

        # x = F.normalize(x, dim=1)

        return x


class Model0Decoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cls_token_count = 32

        self.in_projs = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for _ in range(self.cls_token_count)
        ])

        self.pos_encodings = nn.Parameter(torch.randn(1, self.cls_token_count + 375, out_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=8,
            dim_feedforward=out_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=1,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B x in_dim] e.g. [1 x 64]

        Returns:
            embedding: [B x out_dim x 375]
        """

        B, _ = h.shape

        cls_tokens = torch.stack(
            [proj(h) for proj in self.in_projs],
            dim=1,
        )
        assert cls_tokens.shape == (B, self.cls_token_count, self.out_dim)

        x = F.relu(cls_tokens)

        pos_encodings = self.pos_encodings.expand(B, -1, -1)
        tokens = torch.cat([
            x + pos_encodings[:, 0:self.cls_token_count, :],
            pos_encodings[:, self.cls_token_count:, :],
        ], dim=1)
        assert tokens.shape == (B, self.cls_token_count + 375, self.out_dim), tokens.shape

        x = self.transformer(tokens)
        assert x.shape == (B, self.cls_token_count + 375, self.out_dim)

        x = x[:, self.cls_token_count:, :]
        x = x.permute(0, 2, 1)
        assert x.shape == (B, self.out_dim, 375)

        return x
