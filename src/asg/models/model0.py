import dac
import torch
import torch.nn.functional as F
from torch import nn

NHEAD = 32
DIM_FEEDFORWARD = 1024 * 2
NUM_LAYERS = 1


class Model0(nn.Module):
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
        self.encoder = Model0Encoder(dac_z_dim, h_dim)
        self.decoder = Model0Decoder(h_dim, dac_z_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, 1024, T]

        Returns:
            Tuple of
            h: [B, h_dim]
            z_hat: [B, 1024, T]
        """

        h = self.encoder(z)
        z_hat = self.decoder(h)

        return h, z_hat

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S]

        Returns:
            Tuple of
            h: [B, h_dim]
            z: [B, 1024, T]
        """

        z = self.get_dac_z(x)
        h = self.encoder(z)

        return h, z

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

            return self.dac_model.encoder(x).detach()

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

        x = x[:, 0 : self.cls_token_count, :]  # CLS token outputs
        assert x.shape == (B, self.cls_token_count, self.in_dim)

        x = x.flatten(start_dim=1)
        assert x.shape == (B, self.cls_token_count * self.in_dim)

        x = F.relu(x)

        x = self.out_proj(x)
        assert x.shape == (B, self.out_dim)

        x = F.normalize(x, dim=1)

        return x


class Model0Decoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cls_token_count = 0

        # self.in_projs = nn.ModuleList([
        #     nn.Linear(in_dim, out_dim)
        #     for _ in range(self.cls_token_count)
        # ])
        self.in_proj = nn.Linear(in_dim, out_dim)

        self.pos_encodings = nn.Parameter(
            torch.randn(1, self.cls_token_count + 375, out_dim) * 0.02
        )

        # Causal mask: output tokens can attend to all CLS tokens
        # but only past output tokens
        # T = self.cls_token_count + 375
        # causal_mask = torch.zeros(T, T)
        # future = torch.ones(375, 375).triu(diagonal=1).bool()
        # upper_full = torch.zeros(T, T, dtype=torch.bool)
        # upper_full[self.cls_token_count:, self.cls_token_count:] = future
        # causal_mask.masked_fill_(upper_full, float('-inf'))
        # self.register_buffer('causal_mask', causal_mask)

        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=NUM_LAYERS,
            enable_nested_tensor=False,
        )

        self.memorizer = nn.Parameter(torch.randn(375, out_dim))
        self.normy = nn.LayerNorm(out_dim)
        self.pos_proj = nn.Linear(out_dim, out_dim)
        self.projy2 = nn.Linear(out_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B x in_dim] e.g. [1 x 64]

        Returns:
            embedding: [B x out_dim x 375]
        """

        # x = self.memorizer.expand(h.shape[0], -1, -1)
        # # x = self.normy(x)
        # x = self.projy(x)
        # # x = self.normy(x)
        # return x.permute(0, 2, 1)

        B, in_dim = h.shape

        # cls_tokens = torch.stack(
        #     [proj(h) for proj in self.in_projs],
        #     dim=1,
        # )
        # assert cls_tokens.shape == (B, self.cls_token_count, self.out_dim)
        #
        # x = F.relu(cls_tokens)
        #
        # pos_encodings = self.pos_encodings.expand(B, -1, -1)
        # tokens = torch.cat([
        #     x + pos_encodings[:, 0:self.cls_token_count, :],
        #     pos_encodings[:, self.cls_token_count:, :],
        # ], dim=1)
        # assert tokens.shape == (B, self.cls_token_count + 375, self.out_dim), (
        #     tokens.shape
        # )

        # TODO multiple proj?
        tokens = self.in_proj(h)
        # tokens = h
        assert tokens.shape == (B, self.out_dim)

        tokens = tokens.unsqueeze(1).expand(B, 375, self.out_dim)
        tokens = tokens * self.pos_proj(self.pos_encodings)

        x = self.transformer(
            tokens,
            # mask=self.causal_mask,
        )
        assert x.shape == (B, self.cls_token_count + 375, self.out_dim)

        x = x[:, self.cls_token_count :, :]
        x = x.permute(0, 2, 1)
        assert x.shape == (B, self.out_dim, 375)

        return x
