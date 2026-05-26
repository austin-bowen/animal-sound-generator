import dac
import torch
from torch import nn


class Model0(nn.Module):
    def __init__(self, h_dim: int = 64) -> None:
        super().__init__()

        # Load pretrained model
        model_path = dac.utils.download(model_type="44khz")
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

        # Preprocess: resample + pad to model's expected hop size
        x = x.unsqueeze(1)
        x = self.dac_model.preprocess(x, sample_rate=None)

        with torch.no_grad():
            # z, codes, latents, commitment_loss, codebook_loss = self.dac_model.encode(x)
            z = self.dac_model.encoder(x)

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

        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim))

        self.pos_encodings = nn.Parameter(torch.randn(1, 431 + 1, in_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=8,
            dim_feedforward=in_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=1,
        )

        self.out_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B x in_dim x T] e.g. [1 x 1024 x 431]

        Returns:
            embedding: [B x out_dim]
        """

        B, _, T = x.shape

        x = x.permute(0, 2, 1)
        assert x.shape == (B, T, self.in_dim), x.shape

        cls_in = self.cls_token.expand(B, 1, self.in_dim)
        x = torch.cat([cls_in, x], dim=1)
        assert x.shape == (B, T + 1, self.in_dim)

        x = x + self.pos_encodings

        x = self.transformer(x)
        assert x.shape == (B, T + 1, self.in_dim)

        x = x[:, 0, :]  # CLS token output
        assert x.shape == (B, self.in_dim)

        x = self.out_proj(x)
        assert x.shape == (B, self.out_dim)

        return x


class Model0Decoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_proj = nn.Linear(in_dim, out_dim)

        self.pos_encodings = nn.Parameter(torch.randn(1, 431 + 1, out_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=8,
            dim_feedforward=out_dim,
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
            embedding: [B x out_dim x 431]
        """

        B, _ = h.shape

        x = self.in_proj(h)
        assert x.shape == (B, self.out_dim)

        tokens = torch.zeros(B, 431 + 1, self.out_dim)
        tokens[:, 0, :] += x
        tokens += self.pos_encodings

        x = self.transformer(tokens)
        assert x.shape == (B, 431 + 1, self.out_dim)

        x = x[:, 1:, :]
        x = x.permute(0, 2, 1)
        assert x.shape == (B, self.out_dim, 431)

        return x
